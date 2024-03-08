{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/2003UJAN/Text-Summarizer/blob/main/text-summarizer.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install nltk"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2WAJGIFaJ6KN",
        "outputId": "5235f29c-260c-4b66-9f06-3d8968597f80"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: nltk in /usr/local/lib/python3.10/dist-packages (3.8.1)\n",
            "Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk) (8.1.7)\n",
            "Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk) (1.3.2)\n",
            "Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.10/dist-packages (from nltk) (2023.12.25)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk) (4.66.2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#!/usr/bin/env python\n",
        "# coding: utf-8\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.cluster.util import cosine_distance\n",
        "import numpy as np\n",
        "import networkx as nx\n",
        "\n",
        "def read_article(file_name):\n",
        "    file = open(file_name, \"r\")\n",
        "    filedata = file.readlines()\n",
        "    article = filedata[0].split(\". \")\n",
        "    sentences = []\n",
        "\n",
        "    for sentence in article:\n",
        "        print(sentence)\n",
        "        sentences.append(sentence.replace(\"[^a-zA-Z]\", \" \").split(\" \"))\n",
        "    sentences.pop()\n",
        "\n",
        "    return sentences\n",
        "\n",
        "def sentence_similarity(sent1, sent2, stopwords=None):\n",
        "    if stopwords is None:\n",
        "        stopwords = []\n",
        "\n",
        "    sent1 = [w.lower() for w in sent1]\n",
        "    sent2 = [w.lower() for w in sent2]\n",
        "\n",
        "    all_words = list(set(sent1 + sent2))\n",
        "\n",
        "    vector1 = [0] * len(all_words)\n",
        "    vector2 = [0] * len(all_words)\n",
        "\n",
        "    # build the vector for the first sentence\n",
        "    for w in sent1:\n",
        "        if w in stopwords:\n",
        "            continue\n",
        "        vector1[all_words.index(w)] += 1\n",
        "\n",
        "    # build the vector for the second sentence\n",
        "    for w in sent2:\n",
        "        if w in stopwords:\n",
        "            continue\n",
        "        vector2[all_words.index(w)] += 1\n",
        "\n",
        "    return 1 - cosine_distance(vector1, vector2)\n",
        "\n",
        "def build_similarity_matrix(sentences, stop_words):\n",
        "    # Create an empty similarity matrix\n",
        "    similarity_matrix = np.zeros((len(sentences), len(sentences)))\n",
        "\n",
        "    for idx1 in range(len(sentences)):\n",
        "        for idx2 in range(len(sentences)):\n",
        "            if idx1 == idx2: #ignore if both are same sentences\n",
        "                continue\n",
        "            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)\n",
        "\n",
        "    return similarity_matrix\n",
        "\n",
        "\n",
        "def generate_summary(file_name, top_n=5):\n",
        "    nltk.download(\"stopwords\")\n",
        "    stop_words = stopwords.words('english')\n",
        "    summarize_text = []\n",
        "\n",
        "    # Step 1 - Read text anc split it\n",
        "    sentences =  read_article(file_name)\n",
        "\n",
        "    # Step 2 - Generate Similary Martix across sentences\n",
        "    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)\n",
        "\n",
        "    # Step 3 - Rank sentences in similarity martix\n",
        "    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)\n",
        "    scores = nx.pagerank(sentence_similarity_graph)\n",
        "\n",
        "    # Step 4 - Sort the rank and pick top sentences\n",
        "    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)\n",
        "    print(\"Indexes of top ranked_sentence order are \", ranked_sentence)\n",
        "\n",
        "    for i in range(top_n):\n",
        "      summarize_text.append(\" \".join(ranked_sentence[i][1]))\n",
        "\n",
        "    # Step 5 - Offcourse, output the summarize text\n",
        "    print(\"Summarize Text: \\n\", \". \".join(summarize_text))\n",
        "\n",
        "# let's begin\n",
        "generate_summary( \"msft.txt\", 2)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CF1antSrJ8Ce",
        "outputId": "41dffccd-821a-48aa-ce1a-d84f1e35885e"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "In an attempt to build an AI-ready workforce, Microsoft announced Intelligent Cloud Hub which has been launched to empower the next generation of students with AI-ready skills\n",
            "Envisioned as a three-year collaborative program, Intelligent Cloud Hub will support around 100 institutions with AI infrastructure, course content and curriculum, developer support, development tools and give students access to cloud and AI services\n",
            "As part of the program, the Redmond giant which wants to expand its reach and is planning to build a strong developer ecosystem in India with the program will set up the core AI infrastructure and IoT Hub for the selected campuses\n",
            "The company will provide AI development tools and Azure AI services such as Microsoft Cognitive Services, Bot Services and Azure Machine Learning.According to Manish Prakash, Country General Manager-PS, Health and Education, Microsoft India, said, \"With AI being the defining technology of our time, it is transforming lives and industry and the jobs of tomorrow will require a different skillset\n",
            "This will require more collaborations and training and working with AI\n",
            "That’s why it has become more critical than ever for educational institutions to integrate new cloud and AI technologies\n",
            "The program is an attempt to ramp up the institutional set-up and build capabilities among the educators to educate the workforce of tomorrow.\" The program aims to build up the cognitive skills and in-depth understanding of developing intelligent cloud connected solutions for applications across industry\n",
            "Earlier in April this year, the company announced Microsoft Professional Program In AI as a learning track open to the public\n",
            "The program was developed to provide job ready skills to programmers who wanted to hone their skills in AI and data science with a series of online courses which featured hands-on labs and expert instructors as well\n",
            "This program also included developer-focused AI school that provided a bunch of assets to help build AI skills.\n",
            "Indexes of top ranked_sentence order are  [(0.15083257041122708, ['Envisioned', 'as', 'a', 'three-year', 'collaborative', 'program,', 'Intelligent', 'Cloud', 'Hub', 'will', 'support', 'around', '100', 'institutions', 'with', 'AI', 'infrastructure,', 'course', 'content', 'and', 'curriculum,', 'developer', 'support,', 'development', 'tools', 'and', 'give', 'students', 'access', 'to', 'cloud', 'and', 'AI', 'services']), (0.13161201335715553, ['The', 'company', 'will', 'provide', 'AI', 'development', 'tools', 'and', 'Azure', 'AI', 'services', 'such', 'as', 'Microsoft', 'Cognitive', 'Services,', 'Bot', 'Services', 'and', 'Azure', 'Machine', 'Learning.According', 'to', 'Manish', 'Prakash,', 'Country', 'General', 'Manager-PS,', 'Health', 'and', 'Education,', 'Microsoft', 'India,', 'said,', '\"With', 'AI', 'being', 'the', 'defining', 'technology', 'of', 'our', 'time,', 'it', 'is', 'transforming', 'lives', 'and', 'industry', 'and', 'the', 'jobs', 'of', 'tomorrow', 'will', 'require', 'a', 'different', 'skillset']), (0.11403047674961148, ['Earlier', 'in', 'April', 'this', 'year,', 'the', 'company', 'announced', 'Microsoft', 'Professional', 'Program', 'In', 'AI', 'as', 'a', 'learning', 'track', 'open', 'to', 'the', 'public']), (0.10721749759953525, ['In', 'an', 'attempt', 'to', 'build', 'an', 'AI-ready', 'workforce,', 'Microsoft', 'announced', 'Intelligent', 'Cloud', 'Hub', 'which', 'has', 'been', 'launched', 'to', 'empower', 'the', 'next', 'generation', 'of', 'students', 'with', 'AI-ready', 'skills']), (0.10404298514456578, ['As', 'part', 'of', 'the', 'program,', 'the', 'Redmond', 'giant', 'which', 'wants', 'to', 'expand', 'its', 'reach', 'and', 'is', 'planning', 'to', 'build', 'a', 'strong', 'developer', 'ecosystem', 'in', 'India', 'with', 'the', 'program', 'will', 'set', 'up', 'the', 'core', 'AI', 'infrastructure', 'and', 'IoT', 'Hub', 'for', 'the', 'selected', 'campuses']), (0.10031366655994461, ['That’s', 'why', 'it', 'has', 'become', 'more', 'critical', 'than', 'ever', 'for', 'educational', 'institutions', 'to', 'integrate', 'new', 'cloud', 'and', 'AI', 'technologies']), (0.10001137283486655, ['The', 'program', 'is', 'an', 'attempt', 'to', 'ramp', 'up', 'the', 'institutional', 'set-up', 'and', 'build', 'capabilities', 'among', 'the', 'educators', 'to', 'educate', 'the', 'workforce', 'of', 'tomorrow.\"', 'The', 'program', 'aims', 'to', 'build', 'up', 'the', 'cognitive', 'skills', 'and', 'in-depth', 'understanding', 'of', 'developing', 'intelligent', 'cloud', 'connected', 'solutions', 'for', 'applications', 'across', 'industry']), (0.09916750119894319, ['This', 'will', 'require', 'more', 'collaborations', 'and', 'training', 'and', 'working', 'with', 'AI']), (0.09277191614415067, ['The', 'program', 'was', 'developed', 'to', 'provide', 'job', 'ready', 'skills', 'to', 'programmers', 'who', 'wanted', 'to', 'hone', 'their', 'skills', 'in', 'AI', 'and', 'data', 'science', 'with', 'a', 'series', 'of', 'online', 'courses', 'which', 'featured', 'hands-on', 'labs', 'and', 'expert', 'instructors', 'as', 'well'])]\n",
            "Summarize Text: \n",
            " Envisioned as a three-year collaborative program, Intelligent Cloud Hub will support around 100 institutions with AI infrastructure, course content and curriculum, developer support, development tools and give students access to cloud and AI services. The company will provide AI development tools and Azure AI services such as Microsoft Cognitive Services, Bot Services and Azure Machine Learning.According to Manish Prakash, Country General Manager-PS, Health and Education, Microsoft India, said, \"With AI being the defining technology of our time, it is transforming lives and industry and the jobs of tomorrow will require a different skillset\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "Welcome To Colaboratory",
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}