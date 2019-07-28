package weka.start;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.EnsembleLibrary;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.EnsembleSelection;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.File;
import java.util.Random;

public class KddCupProblemWithEnsembleWeka {

    private static final String PATH_TO_DATA = "/Users/r0c0334/Desktop/Weka/weka-practice/src/main/resources/data/";

    public static Instances loadData(String pathToData, String pathToLabels) throws Exception{
        CSVLoader csvLoader = new CSVLoader();
        csvLoader.setFieldSeparator("\t");
        csvLoader.setNominalAttributes("191-last");
        csvLoader.setSource(new File(pathToData));
        Instances data = csvLoader.getDataSet();
        /*RemoveType removeString = new RemoveType();
        removeString.setOptions(new String[]{"-T","string"});
        removeString.setInputFormat(data);*/
        RemoveUseless removeString = new RemoveUseless();
        removeString.setOptions(new String[]{"-M","99"});
        removeString.setInputFormat(data);
        data = Filter.useFilter(data,removeString);

        ReplaceMissingValues fixMissing = new ReplaceMissingValues();
        fixMissing.setInputFormat(data);
        data = Filter.useFilter(data,fixMissing);

        Discretize discretizeNumeric = new Discretize();
        discretizeNumeric.setOptions(new String[]{"-B","4","-R","first-last"});
        discretizeNumeric.setInputFormat(data);
        data= Filter.useFilter(data,discretizeNumeric);

        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker search = new Ranker();
        search.setOptions(new String[]{"-T","0.001"});

        AttributeSelection attributeSelection = new AttributeSelection();
        attributeSelection.setEvaluator(eval);
        attributeSelection.setSearch(search);
        attributeSelection.SelectAttributes(data);

        data = attributeSelection.reduceDimensionality(data);

        //System.out.println(data.toSummaryString());
        //Load the labels
        csvLoader = new CSVLoader();
        csvLoader.setFieldSeparator("\t");
        csvLoader.setNoHeaderRowPresent(true);
        csvLoader.setNominalAttributes("first-last");
        csvLoader.setSource(new File(pathToLabels));
        Instances labels = csvLoader.getDataSet();
        Instances labeledData = Instances.mergeInstances(data,labels);
        labeledData.setClassIndex(labeledData.numAttributes()-1);
        //System.out.println(labeledData.toSummaryString());
        return labeledData;
    }


    public static double[] evaluate(Classifier model) throws Exception{
        double results[] = new double[4];
        String[] labelFiles = new String[]{"churn","appetency","upselling"};
        double overallScore = 0.0;
        for(int i=0;i<labelFiles.length;i++){
            Instances trainData = loadData(PATH_TO_DATA+ "orange_small_train.data",
                    PATH_TO_DATA+"orange_small_train_"+labelFiles[i]+".labels.txt");
            Evaluation evaluation = new Evaluation(trainData);
            evaluation.crossValidateModel(model,trainData,2,new Random(1));
            results[i] = evaluation.areaUnderROC(trainData.classAttribute().indexOfValue("1"));
            //System.out.println(results[i]);
            overallScore+= results[i];
        }

        System.out.println(overallScore/3);
        return results;
    }
    public static void main(String[] args) throws Exception{

        EnsembleLibrary ensembleLibrary = new EnsembleLibrary();
        ensembleLibrary.addModel("weka.classifiers.trees.J48 -S -C 0.25 -B -M 2");
        //ensembleLibrary.addModel("weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A");
        ensembleLibrary.addModel("weka.classifiers.bayes.NaiveBayes");
        //ensembleLibrary.addModel("weka.classifiers.lazy.IBk");
        //ensembleLibrary.addModel("weka.classifiers.functions.SimpleLogistic");
        //ensembleLibrary.addModel("weka.classifiers.functions.SMO");
        ensembleLibrary.addModel("weka.classifiers.meta.AdaBoostM1");
        //ensembleLibrary.addModel("weka.classifiers.meta.LogitBoost");
        ensembleLibrary.addModel("weka.classifiers.trees.DecisionStump");

        EnsembleLibrary.saveLibrary(new File(PATH_TO_DATA+ "data/ensembleLib.model.xml"),ensembleLibrary,null);
        System.out.println(ensembleLibrary.getModels());

        EnsembleSelection ensembleSelection = new EnsembleSelection();
        ensembleSelection.setOptions(new String[]{
                "-L",PATH_TO_DATA+ "data/ensembleLib.model.xml",
                "-W",PATH_TO_DATA+"esTmp",
                "-B","10",
                "-E","1.0",
                "-V","0.25",
                "-H","100",
                "-I","1.0",
                "-X","2",
                "-P","roc",
                "-A","forward",
                "-R","true",
                "-G","true",
                "-O","true",
                "-S","1",
                "-D","true"
        });
        double[] nBays = evaluate(new NaiveBayes());
        System.out.println("Naive bays result:");
        System.out.println("churn:"+nBays[0]+"\nappetency:"+nBays[1]+"\nupsell:"+nBays[2]);
        System.out.println("=========");
        double[] resES = evaluate(ensembleSelection);
        System.out.println("churn:"+resES[0]+"\nappetency:"+resES[1]+"\nupsell:"+resES[2]);
    }
}
