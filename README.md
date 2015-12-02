# Clockwork RNNs

Clockwork RNNs are a set of simple RNNs that run at different clock speeds but share their hidden states and produce a common output. 

Suppose there are `n` RNNs or modules `{RNN1, . . ., RNNn}` and each RNN is assigned a clock period `T ∈ { T1 , . . . , Tn }`. RNNi is active only if the current time step `t % Ti == 0` Each RNN is internally fully interconnected, but the recurrent connections from RNNj to RNNi exists only if the period Ti is smaller than period Tj. 


Intuitively, the slower RNNs (the ones with larger periods) act as shortcut pathways for information and gradients between large number of timesteps. The faster RNNs (the ones with shorter periods) handle short term, local operations. 

Taking the above logic further, we can imagine that, for a given task, the slower RNN comes in every once in a while, puts in the instructions and then hibernates till it is again active. The slower modules take these instructions and operate on them at every timestep to get the desired output. CWRNN works really well for this setting the of simple generation (which is the premise of the 1st experiment that the CWRNN was trained on in the original paper Section 4.1).

However, when looked from a discriminative perspective, the way connections are made make less sense. The faster modules are seeing the inputs and are making short term decisions or keeping important information in their states. The slower modules that come in later should be able to see these states of faster modules to understand what has occured in the intervening timesteps when it was not active. In other words, we cannot expect a perticular change in input at a perticular timestep in which the slower module becomes active. The observations made by faster module should be summarized by the slower module and stored in the slower module's hidden state. The faster module, due to its shorter memory will forget this information eventually, but can access this since the slower modules have it in their hidden states.

This is much better understood in the example context of Language Modelling. Suppose we have a character level LM, with CWRNN. The faster module, is seeing everything - opening or closing of parenthesis, punctuation, the gender of the subject, the tone of the sentence, the tense of the text / sentences. The slower modules should come in and should be able to retrive all this information, as that is only available to the fastest module. And then the slower module should store it in its hidden state in a some form. It can then make it available for the faster modules at every one of their active timesteps. Seeing one character for every 16 characters in a text sequence can tell nothing about the text.

Hence, we it seems like we should have full duplex interconnections between modules of all speeds.

#### Extensions:

* **Odd - Even Modules**: The fastest modules (usually ones with period 1) are seeing everything. Instead that  can be split up into two modules that are active alternatively. This should help with vanishing gradients. These two modules should be atleast mutually fully interconnected. This can be further extended by having a set of 4 modules such that modules take turns one after that other to be active.
* **Remove the connection from input to slower modules completely**. If the logic stated above is right, then there is little use letting slower modules see any input at all.
* **Multiple RNNs at same speed**: We can have multiple RNNs that are operating at the same speed, and there is a blurry selection between them, then this could be something like hierarchial RNNs.
* **Dynamic Forgetting during training**: Forget hidden state very frequently during inital stages of training, but extend the period of forgetting over time.
* **Various Clock Periods**: 
	* Symmetric: Has given best results so far, within 		the original CWRNN model. Maybe due to the reason 		stated above.
	* Fibonacci (Virahanka)
	* Different Exponential Series
	* Random
* **Dropout**
* **Other Fancy Activations**
	
#### Later Experiments:
Learn the periods.
Initial Idea: Let the RNN emit a gaussian distribution over time periods. This requires a huge recurrent matrix.



## Results

Initial Experiments have shown that:

1. The connection mechanism suggested in the paper, which is from slower to faster works slightly better than full duplex connections, when the number of parameters is kept constant.
2. Adam is a vastly better optimizer than vanilla RmsProp. Need to test against Graves 2013 RmsProp.


After about a 4 days of continous training, the final observations are:

<table>
	<tr>
		<td><b>Clock Periods</b></td>
		<td><b>Number of States (per module)</b></td>
		<td><b>Full Interconnections</b></td>
		<td><b>Effective Size of Recurrent Weight Matrix</b></td>
		<td><b>Avg. Test Loss</b></td>
		<td><b>Remarks</b></td>
	</tr>
	<tr>
		<td>[ 1 ] </td>
		<td>768</td>
		<td>True</td>
		<td>589,824</td>
		<td>1.40</td>
		<td>Baseline SRN</td>
	</tr>
	<tr>
		<td>[ 1, 4, 16, 4, 1 ] </td>
		<td>256</td>
		<td>False</td>
		<td>819,200</td>
		<td>1.30</td>
		<td>Simple Symmetric</td>
	</tr>
	<tr>
		<td>[ 1, 1, 4, 4, 16, 16]</td>
		<td>128</td>
		<td>True</td>
		<td>294,912</td>
		<td>1.50</td>
		<td>This is ideal</td>
	</tr>
	<tr>
		<td>[ 1, 4, 16]</td>
		<td>256</td>
		<td>False</td>
		<td>294,912</td>
		<td>1.45</td>
		<td>Original Paper Proposal</td>
	</tr>
		<tr>
		<td>[ 1, 4, 16]</td>
		<td>128</td>
		<td>True</td>
		<td>147,456</td>
		<td>1.55</td>
		<td></td>
	</tr>
	<tr>
		<td>[ 1, 4, 16]</td>
		<td>256</td>
		<td>True</td>
		<td>589,824</td>
		<td>1.55</td>
		<td>This is ideal</td>
	</tr>	
	<tr>
		<td>[ 1, 4, 16]</td>
		<td>512</td>
		<td>False</td>
		<td>1,179,648</td>
		<td>1.40</td>
		<td>Unstable and Slow</td>
	</tr>
	<tr>
		<td>[1, 1, 2, 3, 5, 8, 13, 22, 13, 8, 5, 3, 2, 1, 1]</td>
		<td>64</td>
		<td>False</td>
		<td>460,800</td>
		<td>0.90</td>
		<td>doutput = 2048. 3M parameters overall. Overfits</td>
	</tr>	
</table>