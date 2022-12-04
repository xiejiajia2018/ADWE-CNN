'''由tokens和labels转化为xml以作打分'''
import pickle as pkl
import xml.etree.ElementTree as ET
from subprocess import check_output

def label_laptop_xml(template, output_fn, from_fn, pred):
    with open(from_fn, 'rb') as f:
        corpus = pkl.load(f)
    
    dom=ET.parse(template)
    root=dom.getroot()
    for zx, sent in enumerate(root.iter("sentence") ) :
        
        tokens=corpus[zx]['tokens']
        lb = pred[zx][:len(tokens)]
        assert len(tokens)==len(lb), print(tokens, lb)

        opins=ET.Element("aspectTerms")
        token_index, jx, tag_on=0, 0, False
        start, end=-1, -1
        for ix, c in enumerate(sent.find('text').text):
            if jx>=len(tokens[token_index]) and token_index+1 < len(tokens):
                another = True
                if c!=' ' and ord(c)!=160:
                    jx = 0
                    token_index += 1
            else:
                another = True if jx==0 else False
            
            tmp_lab = lb[token_index]
            if token_index<len(tokens) and lb[token_index]==1 and another and jx==0:
                if tag_on:
                    end=ix
                    tag_on=False
                    opin=ET.Element("aspectTerm")
                    opin.attrib['term']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)
                start=ix
                tag_on=True
            elif token_index<len(tokens) and lb[token_index]==2 and another and c!=' ' and not tag_on:
                #raise Exception('OI ERROR')
                start=ix
                tag_on=True
            elif token_index<len(tokens) and tag_on and another:
                if c!=' ' and ord(c)!=160 and (lb[token_index]==1 or lb[token_index]==0):                
                    end=ix
                    tag_on=False 
                    opin=ET.Element("aspectTerm")
                    opin.attrib['term']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)
                elif (c==' ' or ord(c)==160) and (token_index+1 ==len(tokens) or lb[token_index+1]!=2):
                    end=ix
                    tag_on=False 
                    opin=ET.Element("aspectTerm")
                    opin.attrib['term']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)

            elif token_index>=len(tokens) and tag_on:
                end=ix
                tag_on=False 
                opin=ET.Element("aspectTerm")
                opin.attrib['term']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)

            jx += 1

        if tag_on:
            tag_on=False
            end=len(sent.find('text').text)
            opin=ET.Element("aspectTerm")
            opin.attrib['term']=sent.find('text').text[start:end]
            opin.attrib['from']=str(start)
            opin.attrib['to']=str(end)
            opins.append(opin)
        sent.append(opins )
    dom.write(output_fn)  



def label_restaurant_xml(template, output_fn, from_fn, pred):
    with open(from_fn, 'rb') as f:
        corpus = pkl.load(f)
    
    dom=ET.parse(template)
    root=dom.getroot()
    for zx, sent in enumerate(root.iter("sentence") ) :
        
        tokens=corpus[zx]['tokens']
        lb = pred[zx][:len(tokens)]
        assert len(tokens)==len(lb)

        opins=ET.Element("Opinions")
        token_index, jx, tag_on=0, 0, False
        start, end=-1, -1
        for ix, c in enumerate(sent.find('text').text):
            if jx>=len(tokens[token_index]) and token_index+1 < len(tokens):
                another = True
                if c!=' ' and ord(c)!=160:
                    jx = 0
                    token_index += 1
            else:
                another = True if jx==0 else False
            
            tmp_lab = lb[token_index]
            if token_index<len(tokens) and lb[token_index]==1 and another and jx==0:
                if tag_on:
                    end=ix
                    tag_on=False
                    opin=ET.Element("Opinion")
                    opin.attrib['target']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)
                start=ix
                tag_on=True
            elif token_index<len(tokens) and lb[token_index]==2 and another and c!=' ' and not tag_on:
                #raise Exception('OI ERROR')
                start=ix
                tag_on=True
            elif token_index<len(tokens) and tag_on and another:
                if c!=' ' and ord(c)!=160 and (lb[token_index]==1 or lb[token_index]==0):                
                    end=ix
                    tag_on=False 
                    opin=ET.Element("Opinion")
                    opin.attrib['target']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)
                elif (c==' ' or ord(c)==160) and (token_index+1 ==len(tokens) or lb[token_index+1]!=2):
                    end=ix
                    tag_on=False 
                    opin=ET.Element("Opinion")
                    opin.attrib['target']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)

            elif token_index>=len(tokens) and tag_on:
                end=ix
                tag_on=False 
                opin=ET.Element("Opinion")
                opin.attrib['target']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)

            jx += 1

        if tag_on:
            tag_on=False
            end=len(sent.find('text').text)
            opin=ET.Element("Opinion")
            opin.attrib['target']=sent.find('text').text[start:end]
            opin.attrib['from']=str(start)
            opin.attrib['to']=str(end)
            opins.append(opin)

        sent.append(opins )
    dom.write(output_fn)  


def calculate_f1(pred, data_dir):
    if 'laptop' in data_dir.lower():
        command="java -cp semeval/eval.jar Main.Aspects semeval/pred.xml semeval/Laptops_Test_Gold.xml"
        command=command.split()
        template="semeval/Laptops_Test_Data_PhaseA.xml"
        label_laptop_xml(template, command[4], data_dir+'/test.pkl', pred)
        acc=check_output(command).split()
        print(acc)
        return float(acc[15])

    elif 'restaurant' in data_dir.lower():
        command="java -cp semeval/A.jar absa16.Do Eval -prd semeval/pred_rest.xml -gld semeval/EN_REST_SB1_TEST.xml.gold -evs 2 -phs A -sbt SB1"
        command=command.split()
        template="semeval/EN_REST_SB1_TEST.xml.A"
        label_restaurant_xml(template, command[6], data_dir+'/test.pkl', pred)
        acc=check_output(command).split()
        print(acc)
        return float(acc[9][10:])

    else:
        raise Exception