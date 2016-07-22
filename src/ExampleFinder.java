import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.ling.LabeledWord;
import edu.stanford.nlp.trees.Tree;
import edu.wlu.cs.levy.CG.KDTree;
import edu.wlu.cs.levy.CG.KeyDuplicateException;
import edu.wlu.cs.levy.CG.KeySizeException;


public class ExampleFinder {
	
	static String getNameForLabel(LabeledWord word)
	{
		return word.word().split("/")[0];
	}
	
	static Set<String> getUniqueNamesForLabels(List<LabeledWord> words)
	{
		HashSet<String> retSet=new HashSet<String>();
		for (int i = 0; i < words.size(); i++) {
			retSet.add(getNameForLabel(words.get(i)));
		}
		
		return retSet;
	}
	
	static List<List<Tree>> exchangeGrammarFailed_Examples(List<Tree> treeset)
	{
		
		HashMap<String, Integer> word_id=new HashMap<String, Integer>();
		System.err.println("exchangeGrammarFailed_Examples: build word_id mapping");
		//first time build word_id
		int wid=0;
		for (int i = 0; i < treeset.size(); i++) {
			Tree queryTree = treeset.get(i);
			Set<String> elements = getUniqueNamesForLabels(queryTree.labeledYield());
			for (String labeledWord : elements) {
				if(!word_id.containsKey(labeledWord))
				{
					word_id.put(labeledWord, wid);
					wid+=1;
				}
			} 
		}
		
		int numWords=word_id.size();
		KDTree<Tree> index=new KDTree<Tree>(numWords);
		System.err.println("numWords: "+numWords);
		System.err.println("exchangeGrammarFailed_Examples: build kdtree");
		int min_words=3;
		int max_words=5;
		double min_signal=0.5;
		Random rand=new Random();
		//second pass build index
		TreeMap<Double,Tree> queryset=new TreeMap<Double,Tree>();
		
		for (int i = 0; i < treeset.size(); i++) {
			Tree queryTree = treeset.get(i);
			Set<String> elements = getUniqueNamesForLabels(queryTree.labeledYield());
			SimpleMatrix label = MotifRNNCoreAnnotations.getGoldClass(queryTree);
			double score=label.elementMaxAbs();
			if(elements.size()<min_words||elements.size()>max_words||score<min_signal)
				continue;
			double[] query=new double[numWords];
			for (String el : elements) {
				query[word_id.get(el)]=1+(rand.nextDouble()-0.5);
			}
			try {
				index.insert(query, queryTree);
				queryset.put(-label.normF()+(rand.nextDouble()-0.5)/10000, queryTree);
			} catch (KeySizeException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (KeyDuplicateException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		

		HashSet<Integer> visited=new HashSet<Integer>(queryset.size());
		List<List<Tree>> exmaples=new LinkedList<List<Tree>>();
		//find a good query tree
		System.err.println("exchangeGrammarFailed_Examples: query near neigbhour");
		  for(Double score : queryset.keySet() ){

			Tree queryTree = queryset.get(score);
			if(visited.contains(queryTree.hashCode()))
				continue;
			visited.add(queryTree.hashCode());
			Set<String> elements = getUniqueNamesForLabels(queryTree.labeledYield());
			double[] query=new double[numWords];
			for (String el : elements) {
				query[word_id.get(el)]=1;
			}
			SimpleMatrix label = MotifRNNCoreAnnotations.getGoldClass(queryTree);
			if(elements.size()<min_words||elements.size()>max_words||label.elementMaxAbs()<min_signal)
				continue;
			
			double maxDiff=0;
			Tree bestConstractSample=null;
			
			try {
				List<Tree> results = index.nearestEuclidean(query, 0.5);//two mismatches
				for (int j = 0; j < results.size(); j++) {
					Tree compareTree = results.get(j);
					if(visited.contains(compareTree.hashCode()))
						continue;
					visited.add(compareTree.hashCode());
					Set<String> elements2 = getUniqueNamesForLabels(compareTree.labeledYield());
					
					if(elements2.size()<min_words||elements2.size()>max_words)
						continue;
					//do intersection
					elements2.retainAll(elements);
					if(elements2.size()<min_words)
						continue;
					SimpleMatrix label2 = MotifRNNCoreAnnotations.getGoldClass(compareTree);
					double diff = label2.minus(label).elementMaxAbs()+label2.minus(label).normF()/100;
					if(diff>maxDiff)
					{
						maxDiff=diff;
						bestConstractSample=compareTree;
					}
				}
			} catch (KeySizeException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	
			if(maxDiff>min_signal)
			{
				LinkedList<Tree> goodExample=new LinkedList<Tree>();
				goodExample.add(queryTree);
				goodExample.add(bestConstractSample);
				
				exmaples.add(goodExample);
				System.err.println(queryTree);
				System.err.println(bestConstractSample);
				System.err.println("-------------------------------");
				if(exmaples.size()>1000) //too slow if we consider all
					break;
			}
		}
		
		return exmaples;
	}
	
	public static List<List<Tree>> predictableExamples(MotifSentimentModel model,List<List<Tree>> interestedExamples)
	{
		List<List<Tree>> examples=new LinkedList<List<Tree>>();
		double allowerr=0.1;
		for (List<Tree> example : interestedExamples) {
			boolean accept=true;
			for (int i = 0; i < example.size(); i++) {
				Tree sample1 = example.get(i);
				SimpleMatrix pred = model.predict(sample1);
				SimpleMatrix trueLabel = MotifRNNCoreAnnotations.getGoldClass(sample1);
				double err=pred.minus(trueLabel).elementMaxAbs();
				if(err>allowerr)
				{
					accept=false;
					break;
				}
			}
			if(accept)
			{
				examples.add(example);
			}
		}
		
		return examples;
	}
	public static void main(String[] args) {
		String trainPath =  "nonExchangable_grammar_dataset.txt";
		List<Tree> trainingTrees = Utils.readTreesWithGoldLabels(trainPath);
		
		List<List<Tree>> examples=exchangeGrammarFailed_Examples(trainingTrees);
		
		
		
		//print sample result
		List<Tree> exmaple1 = examples.get(0);
		for (int i = 0; i < exmaple1.size(); i++) {
			System.out.println(exmaple1.get(i));
		}
		
	  }
	

}
