import cozmo
from cozmo.objects import LightCube1Id, LightCube2Id, LightCube3Id
from PIL import Image
import asyncio
from cozmo.util import degrees, distance_mm

def on_object_tapped(self, event, *, obj, tap_count, tap_duration, **kw):
	robot.say_text("The cube was tapped").wait_for_completed()
	return
def cozmo_program(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True
    robot.world.connect_to_cubes()
    cube1 = robot.world.get_light_cube(LightCube1Id)
    cube2 = robot.world.get_light_cube(LightCube2Id)
    cube3 = robot.world.get_light_cube(LightCube3Id)

    if cube1 is not None:
            cube1.set_lights(cozmo.lights.red_light)
    if cube2 is not None:
            cube2.set_lights(cozmo.lights.green_light)
    if cube3 is not None:
            cube3.set_lights(cozmo.lights.blue_light)
    try:
        robot.say_text("Tap the red block first").wait_for_completed()
        cube1.wait_for_tap(timeout=10)
    except asyncio.TimeoutError:
        robot.say_text("You didn't tap the block in time").wait_for_completed()
        success = False
    finally:
        cube1.set_lights_off()
        if(success):
            robot.say_text("Block 1 was tapped. ").wait_for_completed()
        else:
            robot.say_text("You didn't tap the cube in time").wait_for_completed()
            success = True
    
    try:
        robot.say_text("Tap the green block next")
        cube2.wait_for_tap(timeout=10)
    except asyncio.TimeoutError:
        robot.say_text("You didn't tap the cube in time").wait_for_completed()
        success = False
    finally:
        cube2.set_lights_off()
        if(success):
            robot.say_text("Cube 2 was tapped").wait_for_completed()
        else:
            robot.say_text("You didn't tap the cube in time").wait_for_completed()
            success = True
    
    try:
        robot.say_text("Tap the blue cube last").wait_for_completed()
        cube3.wait_for_tap(timeout=10)
    except asyncio.TimeoutError:
        robot.say_text("You didn't tap the cube in time").wait_for_completed()
        success = False
    finally:
        cube3.set_lights_off()
        if(success):
            robot.say_text("I will now act like a dog").wait_for_completed()
            robot.play_anim_trigger(cozmo.anim.Triggers.DogBark).wait_for_completed()
        else:
            robot.say_text("You didn't tap the cube in time").wait_for_completed()
            success = True
        
        

