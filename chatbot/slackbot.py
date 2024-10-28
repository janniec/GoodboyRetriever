# https://medium.com/@nidhog/how-to-make-a-chatbot-on-slack-with-python-82015517f19c

import os, slackclient, time
import random
from AnswerQuestion import make_slack_answers
import sys
sys.path.append('../../MyModules/')
import Passwords as ps

SOCKET_DELAY = 1

VALET_SLACK_NAME = 'GoodboyRetriever'
VALET_SLACK_TOKEN = ps.LEGACY_BOT_USER_ACCESS_TOKEN
VALET_SLACK_ID = PS.GOODBOYRETRIEVER_APP_ID

valet_slack_client = slackclient.SlackClient(VALET_SLACK_TOKEN)

def is_private(event):
    """Checks if on a private slack channel"""
    channel = event.get('channel')
    return channel.startswith('D')

def post_message(message, channel):
    valet_slack_client.api_call('chat.postMessage', channel=channel, 
                                text=message, as_user=True)

# how the bot is mentioned on slack
def get_mention(user):
    return '<@{user}>'.format(user=user)

valet_slack_mention = get_mention(VALET_SLACK_ID)

# TODO Language Specific
def is_for_me(event):
    """Know if the message is dedicated to me"""
    # check if not my own event
    type = event.get('type')
    if type and type == 'message' and not (event.get('user')==VALET_SLACK_ID):
        if is_private(event):
            return True
        text = event.get('text')
        channel = event.get('channel')
        if valet_slack_mention in text.strip().split():
            return True

def is_yes(message):
    tokens = [word.lower() for word in message.strip().split()]
    return any(g in tokens for g in ['yes', 'sure', 'yeah', 'yup', 'yep', 'definitely']) #add more

def is_no(message):
    tokens = [word.lower() for word in message.strip().split()]
    return any(g in tokens for g in ['no', 'nope', 'nah', 'not'])

def is_question(message):
    tokens = [word.lower() for word in message.strip().split()]
    return (any(g in tokens for g in ['who', 'what', 'when', 'where', 'how',\
                                      'why', 'do', 'am', 'does', 'are',\
                                      'did', 'is', 'can', 'has', 'have', 'will', 'could', 'should', 'would'\
                                     ]) \
            or message.endswith('?'))

def handle_messages(messages_list, channel):
    for m in messages_list:
        post_message(m, channel)
        # time.sleep(SOCKET_DELAY)

# Bot Specific
def run():
    if valet_slack_client.rtm_connect():
        print('[.] Valet de Machin is ON...')
        conversation = []
        while True:
            event_list = valet_slack_client.rtm_read()
            if len(event_list) > 0:
                for event in event_list:
                    print(event)
                    if is_for_me(event):
                        message = event.get('text')
                        user_mention = get_mention(event.get('user'))
                        channel = event.get('channel')
                        if is_question(message):
                            converation = ['?']
                            answer_string, other_answer_string, other_pages_string =\
                            make_slack_answer(message.replace('<@{}>'.format(VALET_SLACK_ID), '').strip())
                            # answer string
                            intro = 'Great question {}'.format(user_mention)
                            closing = 'Did that answer your question?'
                            handle_message([intro, answer_string, closing], channel)
                        elif is_no(message):
                            if conversation == ['?']:
                                # other answer string
                                intro = 'Ok. How about one of these?'
                                closing = 'Did you find the answer you seek?'
                                handle_messages([intro, other_answer_string, closing], channel)
                                conversation.append('no')
                            elif conversation == ['?', 'no']:
                                # other pages string
                                intro = 'Sorry about that. You might find more information here:'
                                handle_messages([intro, other_pages_string], channel)
                                conversation = []
                            else:
                                response = 'I\'m sorry I must have missed something.'
                                handle_messages([response], channel)
                                conversation = []
                        elif is_yes(message):
                            response = 'Awesome! Glad I could help.'
                            handle_messages([response], channel)
                            conversation = []
                        else:
                            response = 'Does not compute. Please ask a question.'
                            handle_messages([response], channel)
            time.sleep(SOCKET_DELAY)        
    else:
        print('[!] Connection to Slack failed! (Have you sourced the environment variables?')
if __name__=='__main__':
    run()
    
# python slackbot.py