import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def self_similarity_matrix(items, metric):
    return np.array([[metric(x, y) for x in items] for y in items])


def draw_ssm_side_by_side(ssm_blocks, ssm_lines, ssm_tokens, representation,
                          song_name='', artist_name='', genre_of_song='undef', save_to_file=False):

    # Create a figure space matrix consisting of 3 columns and 1 row
    fig, ax = plt.subplots(figsize=(39, 13), ncols=3, nrows=1)

    left = 0.125   # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.05  # the amount of width reserved for blank space between subplots
    hspace = 0.1   # the amount of height reserved for white space between subplots

    # This function actually adjusts the sub plots using the above paramters
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # Title of the whole plot
    plt.suptitle("%s - %s (%s)" % (artist_name, song_name, genre_of_song), fontsize=40)

    # Titles of the subplots
    y_title_margin = 1.01
    sub_title_size = 22
    ax[0].set_title("Blocks [" + representation + ']', y=y_title_margin, fontsize=sub_title_size)
    ax[1].set_title("Lines [" + representation + ']', y=y_title_margin, fontsize=sub_title_size)
    ax[2].set_title("Words [" + representation + ']', y=y_title_margin, fontsize=sub_title_size)

    # The color bar [left, bottom, width, height]
    cbar_ax = fig.add_axes([.905, 0.125, .01, 0.751])

    # The actual subplots
    sns.heatmap(data=ssm_blocks, square=True, ax=ax[0], cbar=False)
    sns.heatmap(data=ssm_lines, square=True, ax=ax[1], cbar=False)
    sns.heatmap(data=ssm_tokens, square=True, ax=ax[2], cbar_ax=cbar_ax)

    # Whether to display the plot or save it to a file
    if not save_to_file:
        plt.show()
    else:
        directory = 'SSM/'  # + ('undef' if not genre_of_song else genre_of_song)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + song['_id'] + '.png')
    plt.close('all')


def draw_ssm_encodings_side_by_side(ssm_some_encoding, ssm_other_encoding, ssm_third_encoding,
                                    representation_some, representation_other, representation_third,
                                    song_name='', artist_name='', genre_of_song='undef', save_to_file=False):

    # Create a figure space matrix consisting of 2 columns and 1 row
    fig, ax = plt.subplots(figsize=(39, 13), ncols=3, nrows=1)

    left = 0.125   # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.05  # the amount of width reserved for blank space between subplots
    hspace = 0.1   # the amount of height reserved for white space between subplots

    # This function actually adjusts the sub plots using the above paramters
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # Title of the whole plot
    plt.suptitle("%s - %s (%s)" % (artist_name, song_name, genre_of_song), fontsize=40)

    # Titles of the subplots
    y_title_margin = 1.01
    sub_title_size = 22
    ax[0].set_title("Lines [" + representation_some + ']', y=y_title_margin, fontsize=sub_title_size)
    ax[1].set_title("Lines [" + representation_other + ']', y=y_title_margin, fontsize=sub_title_size)
    ax[2].set_title("Lines [" + representation_third + ']', y=y_title_margin, fontsize=sub_title_size)

    # The color bar [left, bottom, width, height]
    cbar_ax = fig.add_axes([.905, 0.125, .01, 0.751])

    # The actual subplots
    sns.heatmap(data=ssm_some_encoding, square=True, ax=ax[0], cbar=False)
    sns.heatmap(data=ssm_other_encoding, square=True, ax=ax[1], cbar=False)
    sns.heatmap(data=ssm_third_encoding, square=True, ax=ax[2], cbar_ax=cbar_ax)

    # Whether to display the plot or save it to a file
    if not save_to_file:
        plt.show()
    else:
        directory = 'SSM/'  # + ('undef' if not genre_of_song else genre_of_song)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + song['_id'] + '.png')
    plt.close('all')


def draw_ssm_encodings_and_hierarchy(ssm_some_words, ssm_other_words, ssm_some_lines, ssm_other_lines,
                                     representation_some, representation_other, song_name='', artist_name='',
                                     genre_of_song='undef', save_to_file=False):

    # Create a figure space matrix consisting of 2 columns and 2 rows
    fig, ax = plt.subplots(figsize=(26, 26), ncols=2, nrows=2)

    left = 0.125   # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.05  # the amount of width reserved for blank space between subplots
    hspace = 0.1   # the amount of height reserved for white space between subplots

    # This function actually adjusts the sub plots using the above paramters
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # Title of the whole plot
    plt.suptitle("%s - %s (%s)" % (artist_name, song_name, genre_of_song), fontsize=40)

    # Titles of the subplots
    y_title_margin = 1.01
    sub_title_size = 22
    ax[0][0].set_title("Words [" + representation_some + ']', y=y_title_margin, fontsize=sub_title_size)
    ax[0][1].set_title("Words [" + representation_other + ']', y=y_title_margin, fontsize=sub_title_size)
    ax[1][0].set_title("Lines [" + representation_some + ']', y=y_title_margin, fontsize=sub_title_size)
    ax[1][1].set_title("Lines [" + representation_other + ']', y=y_title_margin, fontsize=sub_title_size)

    # The color bar [left, bottom, width, height]
    cbar_ax = fig.add_axes([.905, 0.125, .01, 0.751])

    # The actual subplots
    sns.heatmap(data=ssm_some_words, square=True, ax=ax[0][0], cbar=False)
    sns.heatmap(data=ssm_other_words, square=True, ax=ax[0][1], cbar=False)
    sns.heatmap(data=ssm_some_lines, square=True, ax=ax[1][0], cbar=False)
    sns.heatmap(data=ssm_other_lines, square=True, ax=ax[1][1], cbar_ax=cbar_ax)

    # Whether to display the plot or save it to a file
    if not save_to_file:
        plt.show()
    else:
        directory = 'SSM/'  # + ('undef' if not genre_of_song else genre_of_song)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + artist_name + ' - ' + song_name + '.png')
    plt.close('all')
