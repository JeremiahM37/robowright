using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Robowright;

/// <summary>One robot and engine a test runs on: a row of <see cref="RobotMatrix"/>.</summary>
public sealed record RobotTarget(string Robot, string Backend)
{
    public override string ToString() => $"{Backend}-{Robot}";
}

/// <summary>
/// The robots and engines a test runs on, as rows for xUnit's <c>[MemberData]</c> or NUnit's <c>[TestCaseSource]</c>:
/// ROBOWRIGHT_ROBOT and ROBOWRIGHT_BACKEND (comma lists, as --rw-robot and --rw-backend), else the ones given, else the
/// SO-101 on MuJoCo.
/// </summary>
public static class RobotMatrix
{
    public static IReadOnlyList<RobotTarget> Targets(string[]? robots = null, string[]? backends = null) =>
        (from backend in List("ROBOWRIGHT_BACKEND", backends, "mujoco")
         from robot in List("ROBOWRIGHT_ROBOT", robots, "so101")
         select new RobotTarget(robot, backend)).ToList();

    /// <summary>Each target as a one-element row, for <c>[MemberData(nameof(...))]</c>.</summary>
    public static IEnumerable<object[]> Of(string[]? robots = null, string[]? backends = null) =>
        Targets(robots, backends).Select(t => new object[] { t });

    private static IReadOnlyList<string> List(string env, string[]? given, string fallback)
    {
        var v = Environment.GetEnvironmentVariable(env);
        var list = !string.IsNullOrWhiteSpace(v)
            ? v.Split(',').Select(s => s.Trim()).Where(s => s.Length > 0).ToList()
            : (given ?? Array.Empty<string>()).ToList();
        return list.Count > 0 ? list : new List<string> { fallback };
    }
}

/// <summary>Options for <see cref="RobotTest.RunAsync"/>.</summary>
public sealed class RobotTestOptions
{
    public int Seed { get; set; }

    /// <summary>Run a failed test again this many times on a fresh world (ROBOWRIGHT_RETRIES overrides it).</summary>
    public int Retries { get; set; }

    /// <summary>Where a failed test's trace is saved (ROBOWRIGHT_TRACE_DIR, default robowright-traces).</summary>
    public string? TraceDir { get; set; }
}

/// <summary>A robot test that failed: its message ends with the trace it saved.</summary>
public class RobotTestException : Exception
{
    public string? Trace { get; }

    public RobotTestException(string message, string? trace, Exception inner) : base(message, inner) => Trace = trace;
}

/// <summary>
/// A robot test in any .NET test framework, as Playwright Test runs a web test: a fresh world on the target, retries
/// on a fresh world, and a failure that saves its trace and names it.
/// <code>
/// public static IEnumerable&lt;object[]&gt; Robots => RobotMatrix.Of(new[] { "so101", "panda" });
///
/// [Theory, MemberData(nameof(Robots))]
/// public Task PutsTheCubeInTheBin(RobotTarget target) => RobotTest.RunAsync(rw, target, async rw =>
/// {
///     await rw.Robot.PickAsync("cube");
///     await rw.Robot.PlaceAsync("bin");
///     await rw.Expect("cube", timeout: 2).ToBeInsideAsync("bin");
/// });
/// </code>
/// </summary>
public static class RobotTest
{
    private static readonly Regex Zip = new(@"[^\s""']+\.zip");

    /// <summary>
    /// Run <paramref name="body"/> on a fresh world of <paramref name="target"/> in the server <paramref name="rw"/>.
    /// Returns how many attempts it took (more than one: it was flaky).
    /// </summary>
    public static async Task<int> RunAsync(
        RobowrightClient rw,
        RobotTarget target,
        Func<RobowrightClient, Task> body,
        RobotTestOptions? options = null,
        [CallerMemberName] string test = "")
    {
        options ??= new RobotTestOptions();
        var env = Environment.GetEnvironmentVariable("ROBOWRIGHT_RETRIES");
        var retries = !string.IsNullOrWhiteSpace(env) ? int.Parse(env.Trim()) : options.Retries;
        RobotTestException? last = null;
        for (var attempt = 0; attempt <= retries; attempt++)
        {
            await rw.Robot.LaunchAsync(new RobotLaunchOptions { Robot = target.Robot, Backend = target.Backend, Seed = options.Seed });
            try
            {
                await body(rw);
                if (attempt > 0) Console.WriteLine($"flaky: {test} [{target}] failed {attempt} time{(attempt > 1 ? "s" : "")}, then passed on retry {attempt}");
                return attempt + 1;
            }
            catch (Exception e)
            {
                last = await WithTrace(rw, e, $"{test}[{target}]{(attempt > 0 ? $"-retry{attempt}" : "")}", options);
            }
        }
        throw last!;
    }

    private static async Task<RobotTestException> WithTrace(RobowrightClient rw, Exception e, string name, RobotTestOptions options)
    {
        var dir = options.TraceDir ?? Environment.GetEnvironmentVariable("ROBOWRIGHT_TRACE_DIR") ?? "robowright-traces";
        var path = $"{dir}/{Regex.Replace(name, @"[^A-Za-z0-9._-]+", "_").Trim('_')}.zip";
        try
        {
            Directory.CreateDirectory(dir);
            var saved = await rw.Robot.SaveTraceAsync(new RobotSaveTraceOptions { Path = path });
            var m = Zip.Match(saved);
            var at = m.Success ? m.Value : path;
            return new RobotTestException($"{e.Message}\n\ntrace: {at}\n  robowright show-trace {at}    (or --text for a plain summary)", at, e);
        }
        catch (Exception)
        {
            return new RobotTestException(e.Message, null, e); // no world to save: the failure says why
        }
    }
}
