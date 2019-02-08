package weka.start;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

public class WekaStarter {

    public static void main(String[] args) throws Exception{
        DataSource dataSource = new DataSource("/home/rakesh/Desktop/MLJava/WekaPractice/src/main/resources/zoo.arff");
        Instances data = dataSource.getDataSet();
        System.out.println(data.numInstances()+"\t instances loaded");

        //System.out.println(data.toString());

        //Remove the Animal attribute
        Remove remove = new Remove();
        remove.setOptions(new String[]{"-R","1"});
        remove.setInputFormat(data);
        data = Filter.useFilter(data,remove);
        //System.out.println(data.toString());

        //Learn the attributes that contribute to learning
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker= new Ranker();
        AttributeSelection attributeSelection = new AttributeSelection();
        attributeSelection.setEvaluator(eval);
        attributeSelection.setSearch(ranker);
        attributeSelection.SelectAttributes(data);
        System.out.println("Attributes weight--"+Arrays.toString(attributeSelection.selectedAttributes()));

        //Building a decision tree
        //We selected J48 algo, there other options like IBk, RandomForest, NaiveBays, AdaBoost, Bagging etc
        //just replace the below instantiation to use these algos.
        J48 tree = new J48();
        tree.setOptions(new String[]{"-U"});
        tree.buildClassifier(data);
        /*TreeVisualizer treeVisualizer = new TreeVisualizer(null,tree.graph(), new PlaceNode2());
        JFrame jFrame = new JFrame("Tree Visualizer");
        jFrame.setSize(800,600);
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.getContentPane().add(treeVisualizer);
        jFrame.setVisible(true);
        treeVisualizer.fitToScreen();*/

        //Using trained model for classification process
        Instance instance = new DenseInstance(1.0,new double[]{
                1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,2.0,1.0,0.0,0.0
        });

        Enumeration<Attribute> attributeWekaEnumeration = data.enumerateAttributes();
        ArrayList<Attribute> atts = new ArrayList<Attribute>();

        while (attributeWekaEnumeration.hasMoreElements()){
            atts.add(attributeWekaEnumeration.nextElement());
        }

        Instances dataUnlabeled = new Instances("TestInstances",atts, 0);
        dataUnlabeled.add(instance);
        dataUnlabeled.setClassIndex(dataUnlabeled.numAttributes() - 1);
        double classif = tree.classifyInstance(dataUnlabeled.firstInstance());
        System.out.println(classif);


        //Evaluating classification result
        Classifier classifier = new J48();
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(classifier,data,10, new Random(1), new Object[]{});
        //System.out.println(evaluation.toSummaryString());
        //System.out.println(evaluation.toMatrixString());

    }
}
