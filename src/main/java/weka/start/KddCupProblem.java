package weka.start;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveType;

import java.io.File;
import java.util.Random;

public class KddCupProblem {

    private static final String PATH_TO_DATA = "/home/rakesh/Desktop/MachineLearningJava/WekaPractice/src/main/resources/";

    public static Instances loadData(String pathToData, String pathToLabels) throws Exception{
        CSVLoader csvLoader = new CSVLoader();
        csvLoader.setFieldSeparator("\t");
        csvLoader.setNominalAttributes("191-last");
        csvLoader.setSource(new File(pathToData));
        Instances data = csvLoader.getDataSet();
        RemoveType removeString = new RemoveType();
        removeString.setOptions(new String[]{"-T","string"});
        removeString.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data,removeString);

        //Load the labels
        csvLoader = new CSVLoader();
        csvLoader.setFieldSeparator("\t");
        csvLoader.setNoHeaderRowPresent(true);
        csvLoader.setNominalAttributes("first-last");
        csvLoader.setSource(new File(pathToLabels));
        Instances labels = csvLoader.getDataSet();
        Instances labeledData = Instances.mergeInstances(filteredData,labels);
        labeledData.setClassIndex(labeledData.numAttributes()-1);
        //System.out.println(labeledData.toSummaryString());
        return labeledData;
    }


    public static double[] evaluate(Classifier model) throws Exception{
        double results[] = new double[4];
        String[] labelFiles = new String[]{"churn","appetency","upselling"};
        double overallScore = 0.0;
        for(int i=0;i<labelFiles.length;i++){
            Instances trainData = loadData(PATH_TO_DATA+"orange_small_train.data",
                    PATH_TO_DATA+"orange_small_train_"+labelFiles[i]+".labels.txt");
            Evaluation evaluation = new Evaluation(trainData);
            evaluation.crossValidateModel(model,trainData,5,new Random(1));
            results[i] = evaluation.areaUnderROC(trainData.classAttribute().indexOfValue("1"));
            System.out.println(results[i]);
            overallScore+= results[i];
        }

        System.out.println(overallScore/3);
        return results;
    }
    public static void main(String[] args) throws Exception{

        evaluate(new NaiveBayes());


    }
}
