import os
from openai import OpenAI


def contactStoryTeller(characters=None, story=None, choice=None):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    if story is None:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Generate a 'choose your own adventure' tale in 3-5 parts from the perspective of "
                            "user's selected media characters. Begin by introducing the characters, "
                            "then one segment at a time, offer two choices: go left or go right. Make the two "
                            "choices interesting and fit the story. If given the story so far, continue based "
                            "on the chosen direction."},
                {"role": "user", "content": f"{characters}"}
            ]
        )
    else:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Generate a 'choose your own adventure' tale in 3-5 parts from the perspective of "
                            "user's selected media characters. Begin by introducing the characters, "
                            "then one segment at a time, offer two choices: go left or go right. Make the two "
                            "choices interesting and fit the story. If given the story so far, continue based "
                            "on the chosen direction."},
                {"role": "user", "content": f"{story}\n{choice}"}
            ]
        )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

contactStoryTeller(characters="Characters:[ScoobyDoo,MickyMouse,Bert,Ernie]")

# Only run the below code to create a NEW narrator Narrator = client.beta.assistants.create( instructions="As the
# narrator in a 'choose your own adventure' tale, devise a brief story in 3-5 segments from " "the 3rd person
# point of view of a list of random media characters provided by the user. The first " "segment should always
# introduce the characters. Respond with one portion of the story at a time, " "presenting the user with a binary
# choice: go left or go right. Please provide interesting " "descriptions of the 'go left or go right' choices
# that fit with the story being told.", name="Narrator", model="gpt-3.5-turbo", ) print(Narrator)

# Create Thread
# empty_thread = client.beta.threads.create()
# print(empty_thread)

# Retrieve Thread
# my_thread = client.beta.threads.retrieve('thread_EfJkk1kchi0ITW1hY8N3hrfs')
# print(my_thread)

# Modify Thread
# my_updated_thread = client.beta.threads.update('thread_EfJkk1kchi0ITW1hY8N3hrfs',metadata={
#     "modified": "true",
#     "user":"abc123"
# }
#                                                )
# print(my_updated_thread)
