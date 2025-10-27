# ML_ITMO_lesson_ONE
DGA Domain Detection Challenge

# Overview
Detecting DGA Domains
Welcome to the DGA Domain Detection competition! This competition focuses on detecting algorithmically generated second-level domains (SLDs) within domain names. Your task is to build a machine learning model that can distinguish between legitimate and DGA-generated SLDs.

# Goal
The goal of this competition is to detect algorithmically generated second-level domains (SLDs), which are often used by malware to evade detection. Participants will be provided with a dataset of domain names labeled as DGA (1) or legitimate (0). Note that in some samples, only the SLD is provided, without a top-level domain (TLD). Your task is to build a model that predicts the correct label for each domain in the test set.
Since false positives (predicting a legitimate domain as DGA) can cause serious issues, it is more important to minimize false positives than false negatives.

# Who Can Participate
This competition is designed for students in Using machine learning algorithms to solve cybersecurity problems course— but anyone is welcome to participate and test their skills. Prior experience with classification tasks and Python programming will be helpful, but the problem is approachable with basic ML knowledge.

# Evaluation
Your submissions will be evaluated using the F_beta score with β=0.5
