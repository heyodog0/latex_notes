\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{tikz}
\usetikzlibrary{bayesnet}

\newtheorem{definition}{Definition}
\newtheorem{example}{Example}

\title{Comprehensive Introduction to Bayesian Analysis}
\author{Expert in Bayesian Statistics}
\date{}

\begin{document}

\maketitle

\tableofcontents

\section{Introduction to Probability}

\subsection{What is Probability?}

Probability is a measure of the likelihood that an event will occur. It is a fundamental concept in statistics and forms the basis for Bayesian analysis.

\begin{definition}
Probability is a number between 0 and 1 that expresses the likelihood of an event occurring. A probability of 0 means the event will never occur, while a probability of 1 means the event will always occur.
\end{definition}

\subsection{Basic Probability Rules}

\begin{enumerate}
    \item The probability of any event must be between 0 and 1, inclusive.
    \item The sum of probabilities for all possible outcomes in a sample space must equal 1.
    \item For mutually exclusive events A and B: P(A or B) = P(A) + P(B)
    \item For any event A: P(not A) = 1 - P(A)
\end{enumerate}

\subsection{Conditional Probability}

Conditional probability is the probability of an event occurring given that another event has already occurred.

\begin{definition}
The conditional probability of event A given event B is denoted as P(A|B) and is calculated as:

\[ P(A|B) = \frac{P(A \text{ and } B)}{P(B)} \]
\end{definition}

\section{Bayes' Theorem: The Foundation of Bayesian Analysis}

\subsection{Statement of Bayes' Theorem}

Bayes' Theorem is the cornerstone of Bayesian analysis. It provides a way to update probabilities based on new evidence.

\begin{definition}
Bayes' Theorem states:

\[ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \]

Where:
\begin{itemize}
    \item P(A|B) is the posterior probability of A given B
    \item P(B|A) is the likelihood of B given A
    \item P(A) is the prior probability of A
    \item P(B) is the marginal likelihood of B
\end{itemize}
\end{definition}

\subsection{Components of Bayes' Theorem}

\subsubsection{Prior Probability}

The prior probability, P(A), represents our initial belief about the probability of event A before considering any new evidence.

\subsubsection{Likelihood}

The likelihood, P(B|A), is the probability of observing the evidence B given that hypothesis A is true.

\subsubsection{Marginal Likelihood}

The marginal likelihood, P(B), is the probability of observing the evidence B under all possible hypotheses.

\subsubsection{Posterior Probability}

The posterior probability, P(A|B), is our updated belief about the probability of event A after considering the new evidence B.

\subsection{A Simple Example of Bayes' Theorem}

\begin{example}
Suppose we have a medical test for a rare disease. The test has the following characteristics:
\begin{itemize}
    \item The disease occurs in 1\% of the population (Prior)
    \item The test correctly identifies 95\% of people with the disease (Sensitivity)
    \item The test correctly identifies 90\% of people without the disease (Specificity)
\end{itemize}

If a person tests positive, what is the probability they have the disease?

Let's use Bayes' Theorem:
\begin{itemize}
    \item A: Having the disease
    \item B: Testing positive
\end{itemize}

\begin{align*}
P(A|B) &= \frac{P(B|A) \cdot P(A)}{P(B)} \\[10pt]
&= \frac{P(B|A) \cdot P(A)}{P(B|A) \cdot P(A) + P(B|\text{not A}) \cdot P(\text{not A})} \\[10pt]
&= \frac{0.95 \cdot 0.01}{0.95 \cdot 0.01 + 0.10 \cdot 0.99} \\[10pt]
&\approx 0.0876
\end{align*}

Therefore, even with a positive test result, the probability of having the disease is only about 8.76\%.
\end{example}

This example illustrates the importance of considering both the prior probability and the test characteristics when interpreting test results.

\section{Probability Distributions}

Probability distributions are fundamental to Bayesian analysis. They describe the likelihood of different outcomes for a random variable.

\subsection{Discrete Probability Distributions}

\subsubsection{Bernoulli Distribution}

The Bernoulli distribution models a single binary outcome, such as a coin flip.

\begin{definition}
A Bernoulli distribution has probability mass function:

\[ P(X = x) = p^x(1-p)^{1-x} \]

where x is either 0 or 1, and p is the probability of success.
\end{definition}

\subsubsection{Binomial Distribution}

The Binomial distribution models the number of successes in a fixed number of independent Bernoulli trials.

\begin{definition}
A Binomial distribution has probability mass function:

\[ P(X = k) = \binom{n}{k} p^k(1-p)^{n-k} \]

where n is the number of trials, k is the number of successes, and p is the probability of success on each trial.
\end{definition}

\subsubsection{Poisson Distribution}

The Poisson distribution models the number of events occurring in a fixed interval of time or space.

\begin{definition}
A Poisson distribution has probability mass function:

\[ P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} \]

where λ is the average number of events per interval.
\end{definition}

\subsection{Continuous Probability Distributions}

\subsubsection{Uniform Distribution}

The Uniform distribution assigns equal probability to all values within a given range.

\begin{definition}
A Uniform distribution on the interval [a,b] has probability density function:

\[ f(x) = \begin{cases} 
\frac{1}{b-a} & \text{for } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases} \]
\end{definition}

\subsubsection{Normal (Gaussian) Distribution}

The Normal distribution is a symmetric, bell-shaped distribution that is ubiquitous in statistics due to the Central Limit Theorem.

\begin{definition}
A Normal distribution with mean μ and standard deviation σ has probability density function:

\[ f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \]
\end{definition}

\subsubsection{Exponential Distribution}

The Exponential distribution models the time between events in a Poisson process.

\begin{definition}
An Exponential distribution with rate parameter λ has probability density function:

\[ f(x) = \lambda e^{-\lambda x} \]

for x ≥ 0.
\end{definition}

\section{Bayesian Inference}

Bayesian inference is the process of updating our beliefs about unknown parameters based on observed data.

\subsection{The Bayesian Inference Process}

1. Start with a prior distribution for the unknown parameter(s).
2. Collect data.
3. Calculate the likelihood of the data given the parameter(s).
4. Use Bayes' Theorem to compute the posterior distribution.
5. Interpret and summarize the posterior distribution.

\subsection{Conjugate Priors}

Conjugate priors are prior distributions that, when combined with certain likelihood functions, result in posterior distributions of the same family as the prior.

\begin{example}
For a Binomial likelihood, the Beta distribution is a conjugate prior:

\begin{itemize}
    \item Prior: Beta(α, β)
    \item Likelihood: Binomial(n, p)
    \item Posterior: Beta(α + k, β + n - k)
\end{itemize}

Where k is the number of successes observed in n trials.
\end{example}

\subsection{Bayesian Point Estimation}

Point estimates summarize the posterior distribution with a single value. Common Bayesian point estimates include:

\begin{itemize}
    \item Posterior mean: E[θ|data]
    \item Posterior median: The value that divides the posterior distribution in half
    \item Maximum A Posteriori (MAP): The mode of the posterior distribution
\end{itemize}

\subsection{Credible Intervals}

Credible intervals are the Bayesian analog to frequentist confidence intervals.

\begin{definition}
A 95\% credible interval [a, b] for a parameter θ satisfies:

\[ P(a \leq \theta \leq b | \text{data}) = 0.95 \]
\end{definition}

There are different types of credible intervals:

\begin{itemize}
    \item Equal-tailed credible interval: The interval between the 2.5th and 97.5th percentiles of the posterior distribution.
    \item Highest Posterior Density (HPD) interval: The shortest interval containing 95\% of the posterior probability.
\end{itemize}

\section{Bayesian Model Comparison}

Bayesian model comparison allows us to compare different models based on their ability to explain the observed data.

\subsection{Bayes Factors}

Bayes factors are a way to compare the evidence for two competing models.

\begin{definition}
The Bayes factor for comparing models M1 and M2 is:

\[ BF_{12} = \frac{P(\text{data}|M1)}{P(\text{data}|M2)} \]
\end{definition}

Interpretation of Bayes factors:
\begin{itemize}
    \item BF12 > 1: Evidence favors M1
    \item BF12 < 1: Evidence favors M2
    \item BF12 = 1: No preference for either model
\end{itemize}

\subsection{Model Averaging}

Model averaging allows us to make predictions that account for uncertainty in model selection.

\begin{definition}
The model-averaged posterior distribution of a quantity of interest Δ is:

\[ P(\Delta|\text{data}) = \sum_{i=1}^k P(\Delta|M_i,\text{data})P(M_i|\text{data}) \]

where Mi are the k models being considered.
\end{definition}

\section{Markov Chain Monte Carlo (MCMC) Methods}

MCMC methods are computational techniques for sampling from complex posterior distributions.

\subsection{The Metropolis-Hastings Algorithm}

The Metropolis-Hastings algorithm is a general MCMC method for sampling from a target distribution.

\begin{algorithm}
\caption{Metropolis-Hastings Algorithm}
\begin{algorithmic}
\STATE Initialize θ0
\FOR{t = 1 to T}
    \STATE Propose θ* ~ q(θ*|θt-1)
    \STATE Calculate acceptance ratio: 
    \STATE α = min(1, [p(θ*)q(θt-1|θ*)] / [p(θt-1)q(θ*|θt-1)])
    \IF{U(0,1) < α}
        \STATE θt = θ*
    \ELSE
        \STATE θt = θt-1
    \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}

\subsection{Gibbs Sampling}

Gibbs sampling is a special case of Metropolis-Hastings used when the conditional distributions of each parameter are known.

\begin{algorithm}
\caption{Gibbs Sampling}
\begin{algorithmic}
\STATE Initialize θ0 = (θ1,0, ..., θp,0)
\FOR{t = 1 to T}
    \FOR{i = 1 to p}
        \STATE Sample θi,t ~ p(θi|θ1,t, ..., θi-1,t, θi+1,t-1, ..., θp,t-1, data)
    \ENDFOR
\ENDFOR
\end{algorithmic}
\end{algorithm}

\section{Hierarchical Bayesian Models}

Hierarchical Bayesian models allow us to model complex data structures with multiple levels of variation.

\subsection{Structure of Hierarchical Models}

A typical hierarchical model has the following structure:

\begin{itemize}
    \item Data level: yi ~ f(θi)
    \item Parameter level: θi ~ g(φ)
    \item Hyperparameter level: φ ~ h(η)
\end{itemize}

Where f, g, and h are probability distributions.

\subsection{Advantages of Hierarchical Models}

\begin{itemize}
    \item Allow for partial pooling of information across groups
    \item Can model complex dependencies in data
    \item Reduce overfitting compared to non-hierarchical models
\end{itemize}

\section{Bayesian Decision Theory}

Bayesian decision theory provides a framework for making optimal decisions under uncertainty.

\subsection{Elements of a Decision Problem}

\begin{itemize}
    \item Set of possible actions: A
    \item Set of possible states of nature: Θ
    \item Loss function: L(a, θ) for a ∈ A and θ ∈ Θ
    \item Posterior distribution: p(θ|data)
\end{itemize}

\subsection{Bayes Risk}

The Bayes risk is the expected loss under the posterior distribution.

\begin{definition}
The Bayes risk for action a is:

\[ r(a|\text{data}) = E_{\theta|\text{data}}[L(a,\theta)] = \int L(a,\theta)p(\theta|\text{data})d\theta \]
\end{definition}

\subsection{Bayes Action}

The Bayes action is the action that minimizes the Bayes risk.

\begin{definition}
The Bayes action a* satisfies:

\[ a^* = \arg\min_a r(a|\text{data}) \]
\end{definition}

\section{Conclusion}

This document has provided a comprehensive introduction to the basics of Bayesian analysis. We've covered fundamental concepts of probability, Bayes' Theorem, probability distributions, Bayesian inference, model comparison, computational methods, hierarchical models, and decision theory. These concepts form the foundation for more advanced topics in Bayesian statistics and its applications in various fields.

Bayesian analysis offers a powerful and flexible framework for reasoning under uncertainty, updating beliefs based on evidence, and making decisions in complex environments. As you continue to explore this field, you'll find that these basic principles can be applied to a wide range of problems in science, engineering, finance, and beyond.

\end{document}
