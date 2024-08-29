import pandas as pd

if __name__ == '__main__':
    prompts = [
         ]
    df = pd.DataFrame(prompts, columns=['prompt'])

    df.to_csv('./datasets/handpick_prompts.csv')
