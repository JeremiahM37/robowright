#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace Robowright;

/// <summary>A tool failed (the server answered isError), or the server could not be reached.</summary>
public class RobowrightException : Exception
{
    public RobowrightException(string? tool, string message, Exception? inner = null) : base(message, inner) => Tool = tool;

    /// <summary>The tool that failed, e.g. "robot_pick"; null when the server itself failed.</summary>
    public string? Tool { get; }
}

/// <summary>An expectation that did not hold: robot_expect answered FAIL, and the message says why.</summary>
public class ExpectationException : Exception
{
    public ExpectationException(string message) : base(message) { }
}

/// <summary>How to start the server.</summary>
public sealed class LaunchOptions
{
    /// <summary>The command that starts the server; default robowright mcp (or ROBOWRIGHT_COMMAND, split on spaces).</summary>
    public IReadOnlyList<string>? Command { get; set; }

    /// <summary>Extra environment variables for the server process.</summary>
    public IDictionary<string, string>? Env { get; set; }

    public string? WorkingDirectory { get; set; }

    /// <summary>Whether the server's stderr is copied to this process's (default true) or discarded.</summary>
    public bool InheritStderr { get; set; } = true;
}

/// <summary>
/// robowright for .NET: the same simulated robots, matchers and traces as the Python library, driven through the
/// robowright MCP server (robowright mcp) over stdio. No dependencies.
/// </summary>
/// <example><code>
/// await using var rw = await RobowrightClient.LaunchAsync();
/// await rw.Robot.LaunchAsync(new() { Robot = "panda" });
/// await rw.Robot.PickAsync("cube");
/// await rw.Robot.PlaceAsync("bin");
/// await rw.Expect("cube").ToBeInsideAsync("bin");
/// </code></example>
public sealed class RobowrightClient : IAsyncDisposable
{
    private const string Protocol = "2025-06-18";

    // The argument a matcher takes first, so Expect("cube").ToBeInsideAsync("bin") reads as it does in Python.
    private static readonly Dictionary<string, string> Primary = new()
    {
        ["to_be_inside"] = "container",
        ["to_be_holding"] = "obj",
        ["to_be_near"] = "target",
        ["to_be_above"] = "target",
        ["to_be_touching"] = "other",
        ["to_have_joint"] = "joint",
        ["to_be_visible"] = "camera",
        ["to_be_hidden"] = "camera",
    };

    private readonly Process _proc;
    private readonly ConcurrentDictionary<long, TaskCompletionSource<JsonElement>> _pending = new();
    private readonly SemaphoreSlim _write = new(1, 1);
    private long _next;
    private volatile Exception? _closed;

    private RobowrightClient(Process proc)
    {
        _proc = proc;
        Robot = new RobotTools(this);
        Ros2 = new Ros2Tools(this);
        Sim = new SimTools(this);
    }

    public RobotTools Robot { get; }

    public Ros2Tools Ros2 { get; }

    public SimTools Sim { get; }

    /// <summary>The server's name and version, from its answer to initialize.</summary>
    public IReadOnlyDictionary<string, string> Server { get; private set; } = new Dictionary<string, string>();

    /// <summary>Start the server and connect to it.</summary>
    public static async Task<RobowrightClient> LaunchAsync(LaunchOptions? options = null)
    {
        options ??= new LaunchOptions();
        var command = options.Command;
        if (command == null)
        {
            var env = Environment.GetEnvironmentVariable("ROBOWRIGHT_COMMAND");
            command = string.IsNullOrWhiteSpace(env)
                ? new[] { "robowright", "mcp" }
                : env.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        }
        var psi = new ProcessStartInfo(command[0])
        {
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var a in command.Skip(1)) psi.ArgumentList.Add(a);
        if (options.WorkingDirectory != null) psi.WorkingDirectory = options.WorkingDirectory;
        if (options.Env != null) foreach (var (k, v) in options.Env) psi.Environment[k] = v;
        Process proc;
        try
        {
            proc = Process.Start(psi) ?? throw new InvalidOperationException("no process");
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            throw new RobowrightException(null, $"could not start {string.Join(" ", command)}: {e.Message}", e);
        }
        var inherit = options.InheritStderr;
        _ = Task.Run(async () =>
        {
            string? line;
            while ((line = await proc.StandardError.ReadLineAsync()) != null)
                if (inherit) Console.Error.WriteLine(line);
        });
        var rw = new RobowrightClient(proc);
        _ = Task.Run(rw.ReadAsync);
        try
        {
            var init = await rw.RequestAsync("initialize", new Dictionary<string, object?>
            {
                ["protocolVersion"] = Protocol,
                ["capabilities"] = new Dictionary<string, object?>(),
                ["clientInfo"] = new Dictionary<string, object?> { ["name"] = "robowright-dotnet", ["version"] = "0.1.0" },
            }).ConfigureAwait(false);
            await rw.SendAsync(new Dictionary<string, object?> { ["jsonrpc"] = "2.0", ["method"] = "notifications/initialized" }).ConfigureAwait(false);
            if (init.TryGetProperty("serverInfo", out var info) && info.ValueKind == JsonValueKind.Object)
                rw.Server = info.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.ToString());
            return rw;
        }
        catch
        {
            try { proc.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
            throw;
        }
    }

    /// <summary>Playwright-style assertions on an object name, or robot, gripper, tcp, base; timeout in seconds of simulated time.</summary>
    public Assertions Expect(string subject, double? timeout = null) => new(this, subject, timeout, false);

    /// <summary>Call any tool by its MCP name, e.g. CallAsync("robot_pick", new() { ["object"] = "cube" }), with the server's argument names.</summary>
    public async Task<string> CallAsync(string tool, IDictionary<string, object?>? args = null)
    {
        var arguments = new Dictionary<string, object?>();
        if (args != null) foreach (var (k, v) in args) if (v != null) arguments[k] = v;
        var result = await RequestAsync("tools/call", new Dictionary<string, object?> { ["name"] = tool, ["arguments"] = arguments }).ConfigureAwait(false);
        var text = new List<string>();
        if (result.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.Array)
            foreach (var c in content.EnumerateArray())
                if (c.TryGetProperty("type", out var type) && type.GetString() == "text" && c.TryGetProperty("text", out var t))
                    text.Add(t.GetString() ?? "");
        var output = string.Join("\n", text);
        if (result.TryGetProperty("isError", out var isError) && isError.ValueKind == JsonValueKind.True)
            throw new RobowrightException(tool, output);
        return output;
    }

    internal async Task<string> ExpectCallAsync(string subject, string matcher, object? arg, IDictionary<string, object?>? rest, double? timeout, bool negate)
    {
        var args = new Dictionary<string, object?>();
        if (arg is IDictionary<string, object?> d)
            foreach (var (k, v) in d) args[k] = v;
        else if (arg != null)
        {
            if (!Primary.TryGetValue(matcher, out var key))
                throw new ArgumentException($"{matcher} takes a dictionary of arguments, not {arg}");
            args[key] = arg;
        }
        if (rest != null) foreach (var (k, v) in rest) args[k] = v;
        var output = await CallAsync("robot_expect", new Dictionary<string, object?>
        {
            ["subject"] = subject,
            ["matcher"] = matcher,
            ["args"] = args,
            ["negate"] = negate,
            ["timeout"] = timeout,
        }).ConfigureAwait(false);
        if (!output.StartsWith("PASS", StringComparison.Ordinal)) throw new ExpectationException(output);
        return output;
    }

    /// <summary>Close the worlds and stop the server.</summary>
    public async ValueTask DisposeAsync()
    {
        if (_proc.HasExited) return;
        try
        {
            await CallAsync("robot_close").ConfigureAwait(false);
        }
        catch (RobowrightException)
        {
            // nothing open, or the server is already going
        }
        try { _proc.StandardInput.Close(); } catch (IOException) { }
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        try
        {
            await _proc.WaitForExitAsync(cts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            try { _proc.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
        }
    }

    private async Task SendAsync(Dictionary<string, object?> msg)
    {
        if (_closed != null) throw _closed;
        await _write.WaitAsync().ConfigureAwait(false);
        try
        {
            await _proc.StandardInput.WriteLineAsync(JsonSerializer.Serialize(msg)).ConfigureAwait(false);
            await _proc.StandardInput.FlushAsync().ConfigureAwait(false);
        }
        catch (IOException e)
        {
            throw new RobowrightException(null, "the robowright server is not reading: " + e.Message, e);
        }
        finally
        {
            _write.Release();
        }
    }

    private async Task<JsonElement> RequestAsync(string method, Dictionary<string, object?> parameters)
    {
        var id = Interlocked.Increment(ref _next);
        var tcs = new TaskCompletionSource<JsonElement>(TaskCreationOptions.RunContinuationsAsynchronously);
        _pending[id] = tcs;
        if (_closed != null) _pending.TryRemove(id, out _);
        try
        {
            await SendAsync(new Dictionary<string, object?> { ["jsonrpc"] = "2.0", ["id"] = id, ["method"] = method, ["params"] = parameters }).ConfigureAwait(false);
        }
        catch
        {
            _pending.TryRemove(id, out _);
            throw;
        }
        return await tcs.Task.ConfigureAwait(false);
    }

    private async Task ReadAsync()
    {
        try
        {
            string? line;
            while ((line = await _proc.StandardOutput.ReadLineAsync().ConfigureAwait(false)) != null)
            {
                line = line.Trim();
                if (line.Length == 0) continue;
                JsonElement msg;
                try
                {
                    msg = JsonDocument.Parse(line).RootElement.Clone();
                }
                catch (JsonException)
                {
                    continue; // not a protocol message
                }
                if (msg.ValueKind != JsonValueKind.Object || !msg.TryGetProperty("id", out var idProp) || idProp.ValueKind != JsonValueKind.Number)
                    continue; // a notification
                if (!_pending.TryRemove(idProp.GetInt64(), out var tcs)) continue; // a request from the server we do not serve
                if (msg.TryGetProperty("error", out var err))
                    tcs.TrySetException(new RobowrightException(null, err.TryGetProperty("message", out var m) ? m.ToString() : err.ToString()));
                else
                    tcs.TrySetResult(msg.TryGetProperty("result", out var r) ? r : default);
            }
        }
        catch (IOException)
        {
            // the server went away; reported below
        }
        await _proc.WaitForExitAsync().ConfigureAwait(false);
        _closed = new RobowrightException(null, $"the robowright server exited (code {_proc.ExitCode})");
        foreach (var id in _pending.Keys)
            if (_pending.TryRemove(id, out var tcs)) tcs.TrySetException(_closed);
    }
}
