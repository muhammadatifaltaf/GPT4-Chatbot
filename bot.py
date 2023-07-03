import discord
import responses


async def send_message(message, user_message, is_private):
    try:
        response = responses.get_response(user_message)
        await message.author.send(response) if is_private else await message.channel.send(response)

    except Exception as e:
        print(e)


def run_discord_bot():
    #TOKEN = 'OTUxODc1NTgzMzcyMDM0MDQ5.GbdChN.xtqBNmVPdW1swpGhOy9OMNQ-LtRpOTds052tGs'
    TOKEN = 'MTA4NzcxMTU3MjAwNjI4OTQ4OQ.G6zh9x.FtPYGZ2_GfDBRQ0KcXTssLWnYizzFpA2hDbV2k'
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f'{client.user} is now running!')

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        username = str(message.author)
        user_message = str(message.content)
        channel = str(message.channel)

        print(f'{username} said: "{user_message}" ({channel})')
        
        if channel == "🪅︱join-a-faction":
            if user_message == '?':
                user_message = user_message[1:]
                await send_message(message, user_message, is_private=True)
            else:
                await send_message(message, user_message, is_private=False)

        #if user_message == 'who are you' or 'who are you?':
         #   user_message = user_message
          #  await send_message(message='I am an AI bot', user_message=user_message, is_private=False)

    client.run(TOKEN)