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

What it does not have yet: the pytest plugin's runner features (retries, projects, shards, the HTML report) are
Python's. From .NET you run the world and assert on it in your own test framework, and
`rw.Robot.GenerateTestAsync()` writes the session as a pytest test when you want it in the suite.
