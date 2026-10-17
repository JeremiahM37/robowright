# Traces and steps

Every test can leave a trace: the world's state at every step, the actions taken, the assertions made, and on a
ROS 2 stack the topics and the log. `robowright show-trace` opens it in the viewer, `robowright replay` rebuilds
the run bit for bit, and `robowright codegen` writes a regression test from it.

::: robowright.trace.Trace

::: robowright.steps.step
