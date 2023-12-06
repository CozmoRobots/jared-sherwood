import cozmo
from cozmo.util import degrees, distance_mm

def moveForward(robot, speed, seconds):
    robot.say_text("I will drive forwards").wait_for_completed()
    robot.drive_straight(cozmo.util.distance_mm(100),cozmo.util.speed_mmps(50))
    
def turnSide2Side(robot):
    robot.say_text("I will turn side to side").wait_for_completed()
    robot.turn_in_place(cozmo.util.degrees(-90)).wait_for_completed()
    robot.turn_in_place(cozmo.util.degrees(180)).wait_for_completed()
    robot.turn_in_place(cozmo.util.degrees(-90)).wait_for_completed()
def spin(robot):
    robot.say_text("I will now do a little spin").wait_for_completed()
    robot.turn_in_place(cozmo.util.degrees(360)).wait_for_completed()
def behavior(robot):
    robot.say_text("I will now do a little dance").wait_for_completed()
    robot.play_anim_trigger(cozmo.anim.Triggers.Dance).wait_for_completed()
def moveAlternating(robot,currentX,currentY,x,y):
    while (currentX) < x or (currentY) < y:
        if (currentX) < x:
            currentX = currentX + 1
            robot.drive_straight(cozmo.util.distance_mm(100),cozmo.util.speed_mmps(50)).wait_for_completed()
        if (currentY) < y:
            robot.turn_in_place(cozmo.util.degrees(-90)).wait_for_completed()
            robot.drive_straight(cozmo.util.distance_mm(100),cozmo.util.speed_mmps(50)).wait_for_completed()
            robot.turn_in_place(cozmo.util.degrees(90)).wait_for_completed()
            currentY = currentY + 1 
        if currentX == x or currentY ==  y:
            behavior(robot)
    
            
            
            
