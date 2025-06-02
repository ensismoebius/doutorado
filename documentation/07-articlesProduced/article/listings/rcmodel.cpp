\begin{lstlisting}[language=Python, caption={Code for RC model}]
def convert_to_time(data, tau=5, threshold=0.01):
	spike_time = tau * torch.log(data / (data - threshold))
	return spike_time
\end{lstlisting}
