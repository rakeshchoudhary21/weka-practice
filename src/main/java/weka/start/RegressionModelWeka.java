package weka.start;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.util.Random;

public class RegressionModelWeka {

    public static void main(String[] args) throws Exception{
        CSVLoader csvLoader = new CSVLoader();
        csvLoader.setSource(new File("/home/rakesh/Desktop/MachineLearningJava/WekaPractice/src/main/resources/ENB2012_data.csv"));
        Instances instances = csvLoader.getDataSet();
        System.out.println(instances.numInstances()+"\t instances loaded");

        instances.setClassIndex(instances.numAttributes()-2);
        Remove remove = new Remove();
        remove.setOptions(new String[]{"-R",instances.numAttributes()+""});
        remove.setInputFormat(instances);
        instances = Filter.useFilter(instances,remove);


        LinearRegression model = new LinearRegression();
        model.buildClassifier(instances);
        System.out.println(model);

        //Cross validation
        Evaluation evaluation = new Evaluation(instances);
        evaluation.crossValidateModel(model,instances,10,new Random(1),new String[]{});
        System.out.println(evaluation.toSummaryString());

        //Regression tree
        M5P m5P = new M5P();
        m5P.setOptions(new String[]{""});
        m5P.buildClassifier(instances);
        System.out.println(m5P);
    }
}
