# .NET

Playwright has one API in four languages. robowright's test runner is pytest, but its robots can be driven from
.NET as well: `clients/dotnet` is a client for the robowright MCP server, for .NET 8, with no dependencies. Every
tool an AI agent gets is an async method there, and `Expect` has the same matchers.

```csharp
await using var rw = await RobowrightClient.LaunchAsync(); // starts `robowright mcp`
await rw.Robot.LaunchAsync(new() { Robot = "so101" });
await rw.Robot.PickAsync("cube");
await rw.Expect("gripper").ToBeHoldingAsync("cube");
await rw.Robot.PlaceAsync("bin");
await rw.Expect("cube", timeout: 2).ToBeInsideAsync("bin");
```

A failing expectation throws `ExpectationException`, which fails an xUnit, NUnit or MSTest test the way a
Playwright assertion does. It is not on NuGet yet; its README in
[clients/dotnet](https://github.com/JeremiahM37/robowright/tree/main/clients/dotnet) shows how to reference it.

Snapshot matchers keep their baseline beside the file named as `test_file`, as Python keeps it beside the test:
`await rw.Expect("world").ToMatchStateSnapshotAsync("tidy", new Dictionary<string, object?> { ["test_file"] = "Tests/TidyTests.cs" })`.

In any test framework, `RobotTest.RunAsync` runs a test as Playwright Test does, on a fresh world of each target
`RobotMatrix` names (`ROBOWRIGHT_ROBOT` and `ROBOWRIGHT_BACKEND` override them), with retries on a fresh world (it
returns the attempts it took: more than one, and it was flaky), and a failure that saves its trace and names it in
`RobotTestException`. With xUnit:

```csharp
public class PickTests : IAsyncLifetime
{
    private RobowrightClient rw = null!;
    public async Task InitializeAsync() => rw = await RobowrightClient.LaunchAsync();
    public async Task DisposeAsync() => await rw.DisposeAsync();

    public static IEnumerable<object[]> Robots => RobotMatrix.Of(new[] { "so101", "panda" });

    [Theory, MemberData(nameof(Robots))]
    public Task PutsTheCubeInTheBin(RobotTarget target) => RobotTest.RunAsync(rw, target, async rw =>
    {
        await rw.Robot.PickAsync("cube");
        await rw.Robot.PlaceAsync("bin");
        await rw.Expect("cube", timeout: 2).ToBeInsideAsync("bin");
    }, new RobotTestOptions { Retries = 1 });
}
```

It needs nothing from NuGet. What it does not have yet, of the pytest plugin's runner: trials, projects and
robowright's HTML report; `rw.Robot.GenerateTestAsync()` writes a session as a pytest test when you want those.
