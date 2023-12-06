import cozmo
from cozmo.objects import LightCube1Id, LightCube2Id, LightCube3Id
from PIL import Image
import cv2
from pyzbar.pyzbar import decode
import numpy as np
from cozmo.util import degrees, distance_mm


def cozmo_program(robot: cozmo.robot.Robot,):
    currentScore = 0
    cardsInHand = []
    aceCount = 0
    robot.camera.image_stream_enabled = True
    robot.color_image_enabled = True
    robot.move_lift(-3)
    robot.set_head_angle(degrees(0)).wait_for_completed()
    robot.say_text("Ready to play!").wait_for_completed()
    gameState = True
    while gameState:
        latestImage = robot.world.latest_image
        if latestImage is not None:
            # Convert the Cozmo camera image to a format that OpenCV can work with
            image = np.asarray(latestImage.raw_image)
            imageCV2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Perform QR code detection using pyzbar
            decodedObjects = decode(imageCV2)
            for obj in decodedObjects:
                cardData = obj.data.decode()
                print("QR Code Data:", cardData)
                if cardData is not None:
                    robot.say_text("I have found a card!").wait_for_completed()
                    print("I have found a card")
                    cardInfo = cardData.split()
                    suit = cardInfo[0]
                    name = cardInfo[1]
                    value = int(cardInfo[2])
                    currentScore = currentScore + value
                    robot.say_text("It is a " + name + " of " + suit).wait_for_completed()
                    if name =="Ace":
                        aceCount = aceCount + 1
                    if aceCount > 0:
                        robot.say_text("My hand value is " + str(currentScore) + " or " + str(currentScore + 10)).wait_for_completed()
                    if aceCount == 0:
                        robot.say_text("My hand value is " + str(currentScore)).wait_for_completed()
                         
                    
                    if currentScore > 21:
                        robot.say_text("Oh no, my hand is busted!").wait_for_completed()
                        """ robot.say_text("My current hand is" + str(currentScore) + " points from").wait_for_completed()
                        for c in cardsInHand:
                            robot.say_text(c).wait_for_completed() """
                        gameState = False
                        break
                    if currentScore == 21:
                        robot.say_text("I got 21, I win!").wait_for_completed()
                        """ robot.say_text("My current hand is" + str(currentScore) + " points from").wait_for_completed()
                        for c in cardsInHand:
                            robot.say_text(c).wait_for_completed() """
                        gameState = False
                        break
                    if currentScore == 21 and len(cardsInHand)==2:
                        robot.say_text("I got a black-jack, I win!").wait_for_completed()
                        """ robot.say_text("My current hand is" + str(currentScore) + " points from").wait_for_completed()
                        for c in cardsInHand:
                            robot.say_text(c).wait_for_completed() """
                        gameState = False
                        break
                    if currentScore + 10 == 21 and aceCount>0:
                        robot.say_text("I got 21, I win!").wait_for_completed()
                        """ robot.say_text("My current hand is" + str(currentScore) + " points from").wait_for_completed()
                        for c in cardsInHand:
                            robot.say_text(c).wait_for_completed() """
                        gameState = False
                        break
                    if currentScore >= 17:
                        robot.say_text("I will stay").wait_for_completed()
                        cardsInHand.append((name + " of " + suit))
                        """ robot.say_text("My current hand is" + str(currentScore) + " points from").wait_for_completed()
                        for c in cardsInHand:
                            robot.say_text(c).wait_for_completed() """
                        robot.turn_in_place(cozmo.util.degrees(-45)).wait_for_completed()
                        robot.turn_in_place(cozmo.util.degrees(90)).wait_for_completed()
                        robot.turn_in_place(cozmo.util.degrees(-45)).wait_for_completed()
                        robot.say_text("My cards in hand are: " ).wait_for_completed()
                        for c in cardsInHand:
                            robot.say_text(c).wait_for_completed()
                        #add in a shake to show it is staying, remember to remove the wait for completed and put it on the shake
                        gameState = False
                        break
                    else:
                        
                        cardsInHand.append((name + " of " + suit))
                        robot.say_text("My cards in hand are: " ).wait_for_completed()
                        for c in cardsInHand:
                            robot.say_text(c).wait_for_completed()
                        robot.move_lift(3)
                        robot.say_text("Hit me!").wait_for_completed()
                        robot.move_lift(-6)
                        #robot.say_text("Ready for another card!").wait_for_completed()
    return(0)
                    
                        
                        
                    
              
                        

        

        
        
        

    
    

        
        
    
            
cozmo.run_program(cozmo_program, use_viewer=False, force_viewer_on_top=False)