// A stand-in for `robowright projects --json`.
process.stdout.write(JSON.stringify({ config: "robowright.toml", projects: { nightly: { robot: ["panda", "ur5e"], seed: 5, trace: "on" }, smoke: { robot: "so101" } } }));
