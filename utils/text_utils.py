"""
Utils for generating text in streamlit
"""
import librosa
import streamlit as st
from PIL import Image

from utils import audio_utils


def intro_container():
    st.title(
        'Who spoke when: Choosing the right speaker diarization tool')
    st.markdown(
        'With the increase in applications of automated ***speech recognition systems (ASR)***, '
        'the ability to partition a speech audio stream with multiple speakers into individual'
        ' segments associated with each individual has become a crucial part of understanding '
        'speech data.')
    st.markdown(
        'In this blog post, we will take a look at different open source frameworks for '
        'speaker diarization and provide you with a guide to pick the most suited '
        'one for your use case.')

    st.markdown(
        "Before we get into the technical details, libraries and tools, let's first understand what"
        " speaker diarization is and how it works!")
    st.markdown("---")
    st.header("🗣️ What is speaker diarization?️")

    st.markdown('\n')
    st.markdown(
        'Speaker diarization aims to answer the question of ***"who spoke when"***. In short: diariziation algorithms '
        'break down an audio stream of multiple speakers into segments corresponding to the individual speakers. '
        'By combining the information that we get from diarization with ASR transcriptions, we can '
        'transform the generated transcript into a format which is more readable and interpretable for humans '
        'and that can be used for other downstream NLP tasks.')
    col1_im1, col2_im1, col3_im1 = st.columns([2, 5, 2])

    with col1_im1:
        st.write(' ')

    with col2_im1:
        st.image(Image.open('docs/asr+diar.png'),
                 caption='Workflow of combining the output of both ASR and speaker '
                         'diarization on a speech signal to generate a speaker transcript.',
                 use_column_width=True)

    with col3_im1:
        st.write(' ')

    st.markdown(
        "Let's illustrate this with an example. We have a recording of a casual phone conversation "
        "between two people. You can see what the different transcriptions look like when we "
        "transcribe the conversation with and without diarization.")

    st.markdown('\n')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🎧 Audio recording ")
        st.markdown("<br></br>", unsafe_allow_html=True)
        st.markdown("<br></br>", unsafe_allow_html=True)
        st.markdown("<br></br>", unsafe_allow_html=True)
        audio_data, sampling_frequency = librosa.load('blog_samples/4092.wav')
        st.audio(audio_utils.create_st_audio_virtualfile(audio_data, sampling_frequency))

    with col2:
        st.subheader("❌ Without diarization")
        st.text("I just got back from the gym. oh good.\n"
                "uhuh. How's it going? oh pretty well. It was\n"
                "really crowded today yeah. I kind of\n"
                "assumed everyone would be at the shore.\n"
                "uhhuh. I was wrong. Well it's the\n"
                "middle of the week or whatever so. But\n"
                "it's the fourth of July. mm. So. yeah.\n"
                "People have to work tomorrow. Do you\n"
                "have to work tomorrow? yeah. Did you\n"
                "have off yesterday? Yes. oh that's good.\n"
                "And I was paid too. oh. Is it paid today?\n"
                "No. oh.\n")

    with col3:
        st.subheader('✅ With diarization')
        st.text("A: I just got back from the gym.\n"
                "B: oh good.\n"
                "A: uhhuh.\n"
                "B: How's it going?\n"
                "A: oh pretty well.\n"
                "A: It was really crowded today.\n"
                "B: yeah.\n"
                "A: I kind of assumed everyone would be at \n"
                "the shore.\n"
                "B: uhhuh.\n"
                "A: I was wrong.\n"
                "B: Well it's the middle of the week or\n"
                " whatever so.\n"
                "A: But it's the fourth of July.\n"
                "B: mm.\n"
                "A: So.\n"
                "B: yeah.\n"
                "B: People have to work tomorrow.\n"
                "B: Do you have to work tomorrow?\n"
                "A: yeah.\n"
                "B: Did you have off yesterday?\n"
                "A: Yes.\n"
                "B: oh that's good.\n"
                "A: And I was paid too.\n"
                "B: oh.\n"
                "B: Is it paid today?\n"
                "A: No.\n"
                "B: oh.\n")

    st.markdown(
        "By generating a **speaker-aware transcript**, we can more easily interpret the generated"
        " conversation compared to a generated transcript without diarization. Much neater no? ✨")
    st.caption(
        "But what can I do with these speaker-aware transcripts? 🤔")
    st.markdown(
        "Speaker-aware transcripts can be a powerful tool for analyzing speech data:")
    st.markdown("""
        * We can use the transcripts to analyze individual speaker's sentiment by using **sentiment analysis** on both audio and text transcripts.
        * Another use case is telemedicine where we might identify the **<doctor>** and **<patient>** tags on the transcription to create an accurate transcript and attach it to the patient file or EHR system.
        * Speaker Diarization can be used by hiring platforms to analyze phone and video recruitment calls. This allows them to split and categorize candidates depending on their response to certain questions without having to listen again to the recordings.
        """)
    st.markdown(
        "Now that we've seen the importance of speaker diarization and some of its applications,"
        " it's time to find out how we can implement diarization algorithms.")

    st.markdown("---")
    st.header('📝 The workflow of a speaker diarization system')
    st.markdown(
        "Building robust and accurate speaker diarization is not a trivial task."
        " Real world audio data is messy and complex due to many factors, such"
        " as having a noisy background, multiple speakers talking at the same time and "
        "subtle differences between the speakers' voices in pitch and tone. Moreover, speaker diarization systems often suffer "
        "from **domain mismatch** where a model on data from a specific domain works poorly when applied to another domain.")

    st.markdown(
        "All in all, tackling speaker diarization is no easy feat. Current speaker diarization systems can be divided into two categories: **Traditional systems** and **End-to-End systems**. Let's look at how they work:")
    st.subheader('Traditional diarization systems')
    st.markdown(
        "Those consist of many independent submodules that are optimized individually, namely being:")
    st.markdown("""
            * **Speech detection**: The first step is to identify speech and remove non-speech signals with a voice activity detector (VAD) algorithm.
            * **Speech segmentation**: The output of the VAD is then segmented into small segments consisting of a few seconds (usually 1-2 seconds). 
            * **Speech embedder**: A neural network pre-trained on speaker recognition is used to derive a high-level representation of the speech segments. Those embeddings are vector representations that summarize the voice characteristics (a.k.a voice print).
            * **Clustering**: After extracting segment embeddings, we need to cluster the speech embeddings with a clustering algorithm (for example K-Means or spectral clustering). The clustering produces our desired diarization results, which consists of identifying the number of unique speakers (derived from the number of unique clusters) and assigning a speaker label to each embedding (or speech segment). 
             """)
    col1_im1, col2_im1, col3_im1 = st.columns([2, 5, 2])

    with col1_im1:
        st.write(' ')

    with col2_im1:
        st.image(Image.open('docs/speech_embedding.png'),
                 caption="Process of identifying speaker segments from speech activity embeddings.",

                 use_column_width=True)

    with col3_im1:
        st.write(' ')

    st.subheader('End-to-end diarization systems')
    st.markdown(
        "Here the individual submodules of the traditional speaker diarization system can be replaced by one neural network that is trained end-to-end on speaker diarization.")

    st.markdown('**Advantages**')
    st.markdown(
        '➕ Direct optimization of the network towards maximizing the accuracy for the diarization task. This is in contrast with traditional systems where submodules are optimized individually but not as a whole.')
    st.markdown(
        '➕ Less need to come up with useful pre-processing and post-processing transformation on the input data.')
    st.markdown(' **Disadvantages**')
    st.markdown(
        '➖ More effort needed for data collection and labelling. This is because this type of approach requires speaker-aware transcripts for training. This differs from traditional systems where only labels consisting of the speaker tag along with the audio timestamp are needed (without transcription efforts).')
    st.markdown('➖ These systems have the tendency to overfit on the training data.')

    st.markdown("---")
    st.header('📚 Speaker diarization frameworks')
    st.markdown(
        "As you can see, there are advantages and disadvantages to both traditional and end-to-end diarization systems. "
        "Building a speaker diarization system also involves aggregating quite a few "
        "building blocks and the implementation can seem daunting at first glance. Luckily, there exists a plethora "
        "of libraries and packages that have all those steps implemented and are ready for you to use out of the box 🔥.")
    st.markdown(
        "I will focus on the most popular **open-source** speaker diarization libraries. All but the last framework (UIS-RNN) are based on the traditional diarization approach. Make sure to check out"
        " [this link](https://wq2012.github.io/awesome-diarization/) for a more exhaustive list on different diarization libraries.")

    st.markdown("### 1. [pyannote](https://github.com/pyannote/pyannote-audio)")
    st.markdown(
        "Arguably one of the most popular libraries out there for speaker diarization.\n")
    st.markdown(
        "👉 Note that the pre-trained models are based on the [VoxCeleb datasets](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) which consists of recording of celebrities extracted from YouTube. The audio quality of those recordings are crisp and clear, so you might need to retrain your model if you want to tackle other types of data like recorded phone calls.\n")
    st.markdown(
        "➕ Comes with a set of available pre-trained models for the VAD, embedder and segmentation model.\n")
    st.markdown(
        "➕ The inference pipeline can identify multiple speakers speaking at the same time (multi-label diarization).\n")
    st.markdown(
        "➖ It is not possible to define the number of speakers for the clustering algorithm. This could lead to an over-estimation or under-estimation of the number of speakers if they are known beforehand.")
    st.markdown("### 2. [NVIDIA NeMo](https://developer.nvidia.com/nvidia-nemo)")
    st.markdown(
        "The Nvidia NeMo toolkit has separate collections for Automatic Speech Recognition (ASR), Natural Language Processing (NLP), and Text-to-Speech (TTS) models.\n")
    st.markdown(
        "👉 The models that the pre-trained networks were trained on were trained on [VoxCeleb datasets](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) as well as the [Fisher](https://catalog.ldc.upenn.edu/LDC2004T19) and [SwitchBoard](https://catalog.ldc.upenn.edu/LDC97S62) dataset, which consists of telephone conversations in English. This makes it more suitable as a starting point for fine-tuning a model for call-center use cases compared to the pre-trained models used in pyannote. More information about the pre-trained models can be found [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_diarization/results.html).\n")
    st.markdown(
        "➕ Diarization results can be combined easily with ASR outputs to generate speaker-aware transcripts.\n")
    st.markdown(
        "➕ Possibility to define the number of speakers beforehand if they are known, resulting in a more accurate diarization output.\n")
    st.markdown(
        "➕ The fact that the NeMo toolkit also includes NLP related frameworks makes it easy to integrate the diarization outcome with downstream NLP tasks.\n")
    st.markdown("### 3. [Simple Diarizer](https://github.com/cvqluu/simple_diarizer)")
    st.markdown(
        "A simplified diarization pipeline that can be used for quick testing.\n")
    st.markdown(
        "👉 Uses the same pre-trained models as pyannote.\n")
    st.markdown(
        "➕ Similarly to Nvidia NeMo, there's the option to define the number of speakers beforehand.\n")
    st.markdown(
        "➖ Unlike pyannote, this library does not include the option to fine tune the pre-trained models, making it less suitable for specialized use cases.\n")
    st.markdown(
        "### 4. [SpeechBrain](https://github.com/speechbrain/speechbrain)")
    st.markdown(
        "All-in-one conversational AI toolkit based on PyTorch.\n")
    st.markdown(
        "➕ The SpeechBrain Ecosystem makes it easy to develop integrated speech solutions with systems such ASR, speaker identification, speech enhancement, speech separation and language identification.\n")
    st.markdown(
        "➕ Large amount of pre-trained models for various tasks. Checkout their [HuggingFace page](https://huggingface.co/speechbrain) for more information.\n")
    st.markdown(
        "➕ Contains [comprehensible tutorials](https://speechbrain.github.io/tutorial_basics.html) for various speech building blocks to easily get started.\n")
    st.markdown(
        "➖ Diarization pipeline is still not fully implemented yet but this [might change in the future](https://github.com/speechbrain/speechbrain/issues/1208).")
    st.markdown(
        "### 5. [Kaldi](https://github.com/kaldi-asr/kaldi)")
    st.markdown(
        "Speech recognition toolkit that is mainly targeted towards researchers. It is written in C++ and used to train speech recognition models and decode audio from audio files.\n")
    st.markdown(
        "👉 Pre-trained model is based on the [CALLHOME](https://catalog.ldc.upenn.edu/LDC97S42) dataset which consists of telephone conversation between native English speakers in North America.\n")
    st.markdown(
        "👉 Benefits from large community support. However, mainly targeted towards researchers and less suitable for production ready-solutions.\n")
    st.markdown(
        "➖  Relatively steep learning curve for beginners who don't have a lot of experience with speech recognition systems.\n")
    st.markdown(
        "➖  Not suitable for a quick implementation of ASR/diarization systems. \n")

    st.markdown(
        "### 6. [UIS-RNN](https://github.com/google/uis-rnn)")
    st.markdown(
        "A fully supervised end-to-end diarization model developed by Google.\n")
    st.markdown(
        "👉 Both training and prediction require the usage of a GPU.\n")
    st.markdown(
        "➖ No-pretrained model is available, so you need to train it from scratch on your custom transcribed data.\n")
    st.markdown(
        "➕ Relatively easy to train if you have a large set of pre-labeled data.\n")
    st.markdown("\n")
    st.markdown(
        "Phew 😮‍💨, that's quite some different frameworks! To make it easier to pick the right one for your use case, I've created a simple flowchart that can get you started on picking a suitable library depending on your use case.")

    col1_im2, col2_im2, col3_im2 = st.columns([4, 5, 4])

    with col1_im2:
        st.write(' ')

    with col2_im2:
        st.image(Image.open('docs/flow_chart_diarization_tree.png'),
                 caption='Flowchart for choosing a framework suitable for your diarization use case.',
                 use_column_width=True)

    with col3_im2:
        st.write(' ')


def demo_container():
    st.header('🤖 Demo')
    st.markdown(
        "Alright, you're probably very curious at this point to test out a few diarization techniques "
        "yourself. Below is a demo where you can try a few of the libraries that are mentioned above. "
        "You can try running multiple frameworks at the same time and compare their results by ticking multiple "
        "frameworks and clicking **'Apply'**.\n")
    st.markdown(
        "**Note:** We are including **Nemo** and **pyannote** frameworks since we are operating on a single environment and a dependency conflict can occur when including other frameworks (most diarization frameworks rely "
        "on different and incompatible versions of the same shared packages).")
    st.caption(
        "**Disclaimer**: Keep in mind that due to computational constraints, only the first 30 seconds will be used for diarization when uploading your own recordings. "
        "For that reason, the diarization results may not be as accurate compared to diarization computed on longer recordings. This"
        " is simply due to the fact that the diarization algorithms will have much less data to sample from in order to create meaningful clusters of embeddings for "
        "each speaker. On the other hand, the diarization results from the provided samples are pre-computed on the whole recording (length of around ≈10min) and thus "
        "have more accurate diarization results (only the first 30 seconds are shown).")


def conlusion_container():
    st.title('💬 Conclusions')
    st.markdown("In this blogpost we covered different aspects of speaker diarization.\n")
    st.markdown(
        "👉 First we explained what speaker diarization is and gave a few examples of its different areas of applications.\n")
    st.markdown(
        "👉 We discussed the two main types of systems for implementing diarization system with a solid (high-level) understanding of both **traditional systems** and **end-to-end** systems.")
    st.markdown(
        "👉 Then, we gave a comparison of different diarization frameworks and provided a guide for picking the best one for your use case.")
    st.markdown(
        "👉 Finally, we provided you with an example to quickly try out a few of the diarization libraries.")
