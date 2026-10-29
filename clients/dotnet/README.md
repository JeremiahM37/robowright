# robowright for .NET

The same simulated robots, matchers and traces as the Python library, from C# or any .NET 8 language. The client
starts the robowright MCP server (`robowright mcp`) and talks to it over stdio, so every tool an AI agent gets is an
async method here, and a world you drive from .NET is the same deterministic world a pytest test sees. It has no
dependencies.

It needs robowright itself, with its MCP extra, where the server will run:

```sh
pip install "robowright[mcp] @ git+https://github.com/JeremiahM37/robowright"
```

```csharp
using Robowright;

await using var rw = await RobowrightClient.LaunchAsync();
await rw.Robot.LaunchAsync(new() { Robot = "so101" });
await rw.Robot.PickAsync("cube");
await rw.Expect("gripper").ToBeHoldingAsync("cube");
await rw.Robot.PlaceAsync("bin");
await rw.Expect("cube", timeout: 2).ToBeInsideAsync("bin");
await rw.Expect("gripper").Not.ToBeHoldingAsync("cube");
await rw.Robot.SaveTraceAsync(new() { Path = "trace.zip" });
```

- **Tools** are grouped as `rw.Robot`, `rw.Ros2` and `rw.Sim`, named after the MCP tools in PascalCase with
  `Async` (`robot_move_to` is `MoveToAsync`). As in Playwright for .NET, a tool's required arguments are parameters
  and the rest go in its options object (`new() { Robot = "panda", KeepOpen = true }`). Each returns the tool's
  text: a snapshot of the world, a result or a report.
- **`Expect(subject)`** has the Python matchers (`ToBeInsideAsync`, `ToBeHoldingAsync`, `ToBeNearAsync`,
  `ToHaveJointAsync`, `ToBeVisibleAsync`, ...), `.Not`, and `timeout:` in seconds of simulated time, retried as in
  Python. A matcher's first argument can be given bare (`ToBeInsideAsync("bin")`) or as a dictionary
  (`ToBeNearAsync(new Dictionary<string, object?> { ["target"] = "bin", ["tol"] = 0.02 })`), and a second
  dictionary takes the rest. A failing one throws `ExpectationException`, which fails an xUnit, NUnit or MSTest test.
- **Errors** from a tool throw `RobowrightException`, with `Tool` naming it.
- **`CallAsync(tool, args)`** calls any tool by its MCP name, with the server's own argument names.
- **`RobowrightClient.LaunchAsync(new LaunchOptions { Command = ... })`** picks the server command; default
  `robowright mcp`, or `ROBOWRIGHT_COMMAND`.

It is not on NuGet yet. Reference the project from a checkout
(`<ProjectReference Include="path/to/clients/dotnet/Robowright/Robowright.csproj" />`) or `dotnet pack` it.

`RobotTools.g.cs`, `Ros2Tools.g.cs`, `SimTools.g.cs` and `Assertions.g.cs` are generated from the server's own tool
list by `python scripts/gen_dotnet_client.py`; the Python test suite fails when they fall behind, and runs the
client's own tests (`Robowright.Tests`, no framework, so nothing comes from NuGet) against a stand-in server and the
real one.
