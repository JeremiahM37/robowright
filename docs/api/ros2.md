# ROS 2

Start a launch file, a node or a whole stack and test it as it runs: `rw.ros2_stack` owns the processes and
cleans them up, `world.ros` reads and stubs the ROS graph, and `rw.route` rewrites, records and replays topics
(the counterpart of Playwright's `page.route`).

::: robowright.stack.ros2_stack

::: robowright.stack.Ros2Stack

::: robowright.routes
    options:
      members: [Route, route]

::: robowright.ros2_graph
