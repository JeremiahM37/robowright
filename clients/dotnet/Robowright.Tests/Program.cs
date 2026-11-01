using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading.Tasks;
using Robowright;

if (args.Length > 0 && args[0] == "fake-server")
{
    FakeServer.Run();
    return 0;
}

var passed = 0;
var failed = new List<string>();

async Task Test(string name, Func<Task> body)
{
    try
    {
        await body();
        passed++;
        Console.WriteLine($"ok - {name}");
    }
    catch (Exception e)
    {
        failed.Add(name);
        Console.WriteLine($"not ok - {name}: {e}");
    }
}

static void Check(bool ok, string what)
{
    if (!ok) throw new Exception("check failed: " + what);
}

static async Task<T> Raises<T>(Func<Task> body) where T : Exception
{
    try
    {
        await body();
    }
    catch (T e)
    {
        return e;
    }
    throw new Exception($"expected {typeof(T).Name}, nothing was thrown");
}

static Task<RobowrightClient> Fake()
{
    var self = Environment.ProcessPath!;
    var command = Path.GetFileNameWithoutExtension(self) == "dotnet"
        ? new[] { self, typeof(FakeServer).Assembly.Location, "fake-server" }
        : new[] { self, "fake-server" };
    return RobowrightClient.LaunchAsync(new LaunchOptions { Command = command });
}

static JsonElement Echoed(string output) => JsonDocument.Parse(output).RootElement;

static string Canon(object? o) => JsonSerializer.Serialize(o);

await Test("every tool is a public method of its namespace", async () =>
{
    await using var rw = await Fake();
    Check(rw.Server["name"] == "fake" && rw.Server["version"] == "9", "server");
    var n = new[] { typeof(RobotTools), typeof(Ros2Tools), typeof(SimTools) }
        .Sum(t => t.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly).Length);
    Check(n > 50, $"only {n} tool methods");
    var output = Echoed(await rw.Robot.MoveToAsync(new[] { 0.2, 0, 0.1 }));
    Check(output.GetProperty("name").GetString() == "robot_move_to", output.ToString());
    Check(output.GetProperty("args").ToString() == "{\"target\":[0.2,0,0.1]}", output.ToString());
});

await Test("options reach the server under their own names, and unset ones are left out", async () =>
{
    await using var rw = await Fake();
    var output = Echoed(await rw.Robot.LaunchAsync(new() { Robot = "panda", KeepOpen = true }));
    Check(output.GetProperty("args").ToString() == "{\"robot\":\"panda\",\"keep_open\":true}", output.ToString());
    Check(Echoed(await rw.Robot.WaitAsync(1.5)).GetProperty("args").ToString() == "{\"seconds\":1.5}", "WaitAsync");
});

await Test("a tool error throws RobowrightException naming the tool", async () =>
{
    await using var rw = await Fake();
    var e = await Raises<RobowrightException>(() => rw.Robot.PickAsync("nope"));
    Check(e.Tool == "robot_pick" && e.Message.Contains("no object named"), e.ToString());
});

await Test("Expect passes, fails with the reason, negates with Not, and takes the matcher's first argument bare", async () =>
{
    await using var rw = await Fake();
    Check((await rw.Expect("cube").ToBeInsideAsync("bin")).StartsWith("PASS"), "pass");
    await Raises<ExpectationException>(() => rw.Expect("cube").ToBeInsideAsync("box"));
    Check((await rw.Expect("cube").Not.ToBeInsideAsync("box")).StartsWith("PASS"), "not");
    var output = await rw.Expect("cube", timeout: 2).ToBeInsideAsync(
        new Dictionary<string, object?> { ["container"] = "bin" }, new Dictionary<string, object?> { ["min_fraction"] = 0.5 });
    Check(output[6..] == Canon(new Dictionary<string, object?>
    {
        ["subject"] = "cube",
        ["matcher"] = "to_be_inside",
        ["args"] = new Dictionary<string, object?> { ["container"] = "bin", ["min_fraction"] = 0.5 },
        ["negate"] = false,
        ["timeout"] = 2,
    }), output);
    Check(!Echoed((await rw.Expect("cube").ToBeInsideAsync("bin"))[6..]).TryGetProperty("timeout", out _), "no timeout sent");
});

await Test("a matcher with no first argument refuses a bare value", async () =>
{
    await using var rw = await Fake();
    await Raises<ArgumentException>(() => rw.Expect("cube").ToBeAtRestAsync(3));
});

await Test("calls in flight fail, not hang, when the server dies", async () =>
{
    var rw = await Fake();
    var e = await Raises<RobowrightException>(() => rw.CallAsync("robot_crash"));
    Check(e.Message.Contains("exited (code 3)"), e.Message);
    await Raises<RobowrightException>(() => rw.Robot.SnapshotAsync());
    await rw.DisposeAsync();
});

await Test("a command that does not exist fails launch", async () =>
{
    var e = await Raises<RobowrightException>(() =>
        RobowrightClient.LaunchAsync(new LaunchOptions { Command = new[] { "robowright-no-such-command" } }));
    Check(e.Message.Contains("could not start"), e.Message);
});

await Test("RobotMatrix names every robot on every engine", async () =>
{
    await Task.CompletedTask;
    if (Environment.GetEnvironmentVariable("ROBOWRIGHT_ROBOT") is { Length: > 0 }) return; // the environment's own list wins
    var t = RobotMatrix.Targets(new[] { "panda", "ur5e" }, new[] { "mujoco", "drake" });
    Check(string.Join(",", t) == "mujoco-panda,mujoco-ur5e,drake-panda,drake-ur5e", string.Join(",", t));
    Check(RobotMatrix.Targets().Single() == new RobotTarget("so101", "mujoco"), "default");
    Check(RobotMatrix.Of().Single()[0] is RobotTarget, "rows");
});

await Test("RobotTest.RunAsync launches the target, names a failure's trace, and retries a flaky test", async () =>
{
    await using var rw = await Fake();
    var target = new RobotTarget("panda", "mujoco");
    var attempts = await RobotTest.RunAsync(rw, target, async rw =>
    {
        var launched = await rw.CallAsync("robot_last_launch");
        Check(launched.Contains("\"robot\":\"panda\"") && launched.Contains("\"seed\":3"), launched);
    }, new RobotTestOptions { Seed = 3 });
    Check(attempts == 1, "one attempt");
    var e = await Raises<RobotTestException>(() => RobotTest.RunAsync(rw, target, rw => rw.Expect("cube").ToBeInsideAsync("box"), test: "Misses"));
    Check(e.Message.Contains("trace: robowright-traces/Misses_mujoco-panda.zip"), e.Message);
    Check(e.Message.Contains("robowright show-trace robowright-traces/Misses_mujoco-panda.zip"), e.Message);
    Check(e.InnerException is ExpectationException, "the matcher's own error inside");
    var flaky = await RobotTest.RunAsync(rw, target, rw => rw.CallAsync("robot_flaky"), new RobotTestOptions { Retries = 1 });
    Check(flaky == 2, "passed on its retry");
});

if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("ROBOWRIGHT_COMMAND")))
{
    await Test("RobotTest against the real server: a pick and place passes, and a failure's trace is saved", async () =>
    {
        await using var rw = await RobowrightClient.LaunchAsync(new LaunchOptions { InheritStderr = false });
        var target = new RobotTarget("so101", "mujoco");
        await RobotTest.RunAsync(rw, target, async rw =>
        {
            await rw.Robot.PickAsync("cube");
            await rw.Robot.PlaceAsync("bin");
            await rw.Expect("cube", timeout: 2).ToBeInsideAsync("bin");
        });
        var dir = Path.Combine(Path.GetTempPath(), $"rw-dotnet-{Environment.ProcessId}");
        var e = await Raises<RobotTestException>(() => RobotTest.RunAsync(
            rw, target, rw => rw.Expect("cube", timeout: 0.2).ToBeInsideAsync("bin"), new RobotTestOptions { TraceDir = dir }, "LeavesTheCube"));
        Check(e.Message.Contains("bin spans"), e.Message);
        Check(e.Trace != null && File.Exists(e.Trace), e.Message);
    });

    await Test("against the real server: pick the cube, hold it, and fail to find it in the bin", async () =>
    {
        await using var rw = await RobowrightClient.LaunchAsync(new LaunchOptions { InheritStderr = false });
        Check(rw.Server["name"] == "robowright", "server");
        Check((await rw.Robot.LaunchAsync()).Contains("cube: red box"), "launch");
        await rw.Robot.PickAsync("cube");
        Check((await rw.Expect("gripper").ToBeHoldingAsync("cube")).StartsWith("PASS"), "holding");
        var e = await Raises<ExpectationException>(() => rw.Expect("cube", timeout: 0.2).ToBeInsideAsync("bin"));
        Check(e.Message.Contains("bin spans"), e.Message);
        var err = await Raises<RobowrightException>(() => rw.Robot.PickAsync("nope"));
        Check(err.Message.Contains("scene has"), err.Message);
    });
}
else
{
    Console.WriteLine("ok - against the real server # SKIP set ROBOWRIGHT_COMMAND to run it");
}

Console.WriteLine($"# pass {passed}");
Console.WriteLine($"# fail {failed.Count}");
return failed.Count == 0 ? 0 : 1;

/// <summary>
/// A stand-in for robowright mcp: answers the MCP handshake and tools/call from a script, so the client's own code is
/// tested without Python. Each call is echoed back, so tests can see what was sent.
/// </summary>
internal static class FakeServer
{
    private static void Send(object m)
    {
        Console.Out.WriteLine(JsonSerializer.Serialize(m));
        Console.Out.Flush();
    }

    private static Dictionary<string, object?> Text(JsonElement id, string text, bool error)
    {
        var result = new Dictionary<string, object?> { ["content"] = new[] { new Dictionary<string, object?> { ["type"] = "text", ["text"] = text } } };
        if (error) result["isError"] = true;
        return new Dictionary<string, object?> { ["jsonrpc"] = "2.0", ["id"] = id, ["result"] = result };
    }

    private static string? lastLaunch;
    private static int flaky;

    public static void Run()
    {
        Console.Out.WriteLine("not a protocol line");
        string? line;
        while ((line = Console.In.ReadLine()) != null)
        {
            var m = JsonDocument.Parse(line).RootElement;
            var method = m.TryGetProperty("method", out var me) ? me.GetString() : null;
            if (method == "initialize")
            {
                Send(new Dictionary<string, object?> { ["jsonrpc"] = "2.0", ["method"] = "notifications/message", ["params"] = new { level = "info", data = "hello" } });
                Send(new Dictionary<string, object?>
                {
                    ["jsonrpc"] = "2.0",
                    ["id"] = m.GetProperty("id"),
                    ["result"] = new { protocolVersion = m.GetProperty("params").GetProperty("protocolVersion").GetString(), capabilities = new { }, serverInfo = new { name = "fake", version = "9" } },
                });
            }
            else if (method == "tools/call")
            {
                var p = m.GetProperty("params");
                var name = p.GetProperty("name").GetString()!;
                var args = p.GetProperty("arguments");
                var id = m.GetProperty("id");
                if (name == "robot_crash") Environment.Exit(3);
                if (name == "robot_launch") lastLaunch = args.GetRawText();
                if (name == "robot_last_launch")
                {
                    Send(Text(id, lastLaunch ?? "null", false));
                    continue;
                }
                if (name == "robot_flaky")
                {
                    flaky++;
                    Send(Text(id, $"attempt {flaky}", flaky == 1));
                    continue;
                }
                if (name == "robot_pick" && args.TryGetProperty("object", out var o) && o.GetString() == "nope")
                    Send(Text(id, "KeyError: no object named 'nope'", true));
                else if (name == "robot_expect")
                {
                    var inBin = args.GetProperty("args").TryGetProperty("container", out var c) && c.GetString() == "bin";
                    var ok = inBin != args.GetProperty("negate").GetBoolean();
                    Send(Text(id, (ok ? "PASS" : "FAIL") + ": " + args.GetRawText(), false));
                }
                else
                    Send(Text(id, JsonSerializer.Serialize(new { name, args }), false));
            }
        }
    }
}
