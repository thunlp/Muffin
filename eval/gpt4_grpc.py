import openai
import time

NUM_SECONDS_TO_SLEEP = 0.5


# 'gpt-4-0314'
def get_eval(model, content: str,
             chat_gpt_system='You are a helpful and precise assistant for checking the quality of the answer.',
             max_tokens: int=256,
             fail_limit=100):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{
                    'role': 'system',
                    'content': chat_gpt_system
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response['choices'][0]['message']['content']