# word-lever vs Char-level Ngram Language Models

N-Gram as the most intuitive computational language model can be implemented in the word-level or the char-leverl while having freedom-of-choice in terms of the N variable. In this project a fair comparision of uni-gram and bi-gram models in the word and charachter level. <br><br><br>
![Ngram gif](https://github.com/SajjadPSavoji/Ngram_Language_Model/blob/main/ezgif.com-gif-maker.gif)
## Results

<table>
<tr>
<td> Model </td> <td> Abstraction Level</td> <td>Data Partition</td> <td> F1-macro </td> <td> F1-micro </td> <td> acc </td> <td> prec </td> <td> recall </td>
</tr>

<tr>
<td> unigram </td> <td> word</td> <td>train</td> <td> 0.878 </td> <td> 0.878 </td> <td> 0.878 </td> <td> 0.769 </td> <td> 0.955 </td>
</tr>

<tr>
<td> unigram </td> <td> word</td> <td>test</td> <td> 0.642 </td> <td> 0.644 </td> <td> 0.644 </td> <td> 0.556 </td> <td> 0.817 </td>
</tr>

<tr>
<td> unigram </td> <td> char</td> <td>train</td> <td> 0.557 </td> <td> 0.590 </td> <td> 0.590 </td> <td> 0.517 </td> <td> 0.379 </td>
</tr>

<tr>
<td> unigram </td> <td> char</td> <td>test</td> <td> 0.539 </td> <td> 0.558 </td> <td> 0.558 </td> <td> 0.477 </td> <td> 0.372 </td>
</tr>



<tr>
<td> bigram </td> <td> word</td> <td>train</td> <td> 0.980 </td> <td> 0.981 </td> <td> 0.981 </td> <td> 0.960 </td> <td> 0.995 </td>
</tr>

<tr>
<td> bigram </td> <td> word</td> <td>test</td> <td> 0.539 </td> <td> 0.567 </td> <td> 0.567 </td> <td> 0.497 </td> <td> 0.950 </td>
</tr>

<tr>
<td> bigram </td> <td> char</td> <td>train</td> <td> 0.679 </td> <td> 0.687 </td> <td> 0.687 </td> <td> 0.620 </td> <td> 0.650 </td>
</tr>

<tr>
<td> bigram </td> <td> char</td> <td>test</td> <td> 0.640 </td> <td> 0.648 </td> <td> 0.648 </td> <td> 0.587 </td> <td> 0.590 </td>
</tr>

</table>


## Resources

- [Char-level language models](https://towardsdatascience.com/character-level-language-model-1439f5dd87fe)
- [word-level language models](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/)
- -[N-Gram language model](https://www.rev.com/blog/resources/understanding-n-gram-language-models)
- [Language modeling with nltk](https://medium.com/swlh/language-modelling-with-nltk-20eac7e70853)

