def writeFile(title, story):
    # This will be used to update the story so far to keep track for the assistant.
    path = f'C:\\Users\\sherw\\PycharmProjects\\RoboticsFinal\\Stories\\{title}'
    file = open(path, "a+")
    file.seek(0)
    data = file.read(100)
    if len(data)>0:
        file.write("\n")
    file.write(story)
    file.close()

    file = open(path, "r")
    content = file.read()
    print(content)
    file.close()
    return


# readFile("newStory","this is some text to try \n"
#                     "Here is some more text to add now!")

