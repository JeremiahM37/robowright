from robowright import expect


def test_lifts_the_cube(robot, scene):
    robot.pick(scene["cube"])
    expect(scene["cube"]).to_be_above(0.04)


def test_reaches_the_shelf(robot, scene):
    robot.pick(scene["cube"])
    expect(scene["cube"]).to_be_above(0.5, timeout=1.0)  # fails: the arm only lifts it a little
