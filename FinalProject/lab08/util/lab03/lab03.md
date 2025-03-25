# Prompt Engineering Process

First, after reading the instructions, I spent time thinking about my experiences with DND and what makes a game fun for me. I really value iterative work, so I am excited for this project.

So, when providing a message to the model, I first tried this prompt "'You should be creative and witty, like a human gamer who is passionate about DND.", because that is how I want the model to behave, like an interesting and real DM. 

To start, I set the temperature to 2.0, because I believed this would make the model exceptionally creative and unconventional. Then I set the max tokens to 100, just as a starting point, because that seems like an appropriate response length, but I am prepared to adjust it to my preferences going forward. Unfrotunately, the first attempt was not recorded because of my code error, but all further attempts are in the text file.

After my first test, I loved the story the AI weaved, as it was very descriptive, so for now I am happy with my prompt. However, I thought the responses were a little too wordy and long, so I adjusted the max tokens to 50, so that the model will generate less words. After running with the change, I was satsified with the amount of content in the responses.

Next, I wanted to make sure the AI doesn't repeat itself too much. I research different option parameters for the model, and found that frequency_penalty could help make my AI repeat fewer phrses word-for-word. So, I added in that option and set it to 1.5 out of 2. I ran the script once more, and I didn't notice repeated phrases, or at least they weren't distracting, so I concluded I found a good solution.

Then, I looked into the option of presence_penalty, encouraging the model to choose (or not choose) words that already appear. My thinking is that I want the AI to have a good "memory" and recall and repeat some past story elements, while still not being too repeptive, as that would take away from my previous intentional choices. So I set the presence_penalty to -1 for my next try.

After this, I had a fun experience creating an elf character on the next run, and I think the model's amount of repetition was appropiate and satsifying. So, I felt good about the parameters I gave it. I think I picked a great base model because it seems well-suited to gameplay, encouraging me to have a fun adventure.





