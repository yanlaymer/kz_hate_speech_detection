# -*- coding: utf-8 -*-
import click
import pandas as pd
import string
import re


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def preprocess(input_filepath:str, output_filepath:str):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    df = pd.read_csv(input_filepath)
    print("preprocess started")

    df.text = df.text.str.lower()
    df.text.dropna(inplace=True)
    df.text = df.text.apply(remove_extra_whitespace)
    df.text = df.text.apply(remove_punctuation)
    df.text = df.text.apply(remove_emojis)

    df.to_csv(output_filepath)
    print("preprocessed data stored")


def remove_extra_whitespace(text):
    return " ".join(text.split())


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_punctuation(text):
    return "".join([i for i in text if i not in string.punctuation])


if __name__ == '__main__':
    preprocess()
