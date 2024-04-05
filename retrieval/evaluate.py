import argparse
import json
from collections import defaultdict

class evaluation():
    def __init__(self, qrel_path, result_path, reverse=True):
        self.qrel_path = qrel_path
        self.result_path = result_path
        self.qrels = self.read_qrels(self.qrel_path)
        self.results = self.read_results(self.result_path)
        self.reverse = reverse

    def read_qrels(self, qrel_path):#qrel为正确答案
        qrels = {} 
        with open(qrel_path) as f:
            for line in f:
                line = line.strip().split('\t')
                qid = line[0]
                pid = line[2]
                if qid not in qrels:
                    qrels[qid] = []
                qrels[qid].append(pid)
                print('1')
        #print(qrels)
        return qrels

    def read_results(self, result_path):#result为模型生成的结果
        results = {}
        with open(result_path) as f:
            for line in f:
                line = line.strip().split('\t')
                qid, pid, score = line
                if qid not in results:
                    results[qid] = [(pid, float(score))]
                else:
                    results[qid].append((pid, float(score)))
                #for pid, score in zip(pids, scores):
                #    results[qid].append((pid, float(score)))
        return results

    def eval(self, topk,test_data):
        outlist=[]
        mrr = 0
        count = 0
        num=0
        no_judged = 0
        for qid in self.results:
            if qid not in self.qrels:
                no_judged += 1
                continue
            res = self.results[qid] #[('122', 268.7765), ('120', 276.55438), ('119', 540.804)]
            temp=[]
            outcome=self.results[qid]
            for i in range(0,len(self.results[qid])):
                a=outcome[i]
                temp.append(a[0])
            count=count+1
            out_dict = {'lable': self.qrels[qid], 'pre':temp,'result':set(self.qrels[qid]) <= set(temp)}
            outlist.append(out_dict)
            if set(self.qrels[qid]) <= set(temp):#
                num=num+1
        return count,num,float(num)/count,outlist

        #     #print(res)   #[('122', 268.7765), ('120', 276.55438)]
        #     # break
        #     ar = 0
        #     sorted_res = sorted(res, key = lambda x:x[1], reverse=self.reverse)#对列表 res 中的元素按照它们的第二个值进行排序,
        #     # print(sorted_res[:10])  #reverse 的值为 True，则会按降序排列，如果为 False，则按升序排列，这里为升序排序
        #     for i, ele in enumerate(sorted_res):
        #         pid = ele[0]
        #         if i >= topk:
        #             break
        #         print(self.qrels[qid])  #正确答案['122']['12']['245','26']...
        #         print(pid)  #122 12
        #         #if pid ==self.qrels[qid]:
        #         if pid in self.qrels[qid]:
        #             ar = 1.0 / (i+1)
        #             break
        #         if pid in self.qrels[qid]:
        #             num+=1
        #     mrr += ar
        #     if ar > 0:
        #         count += 1
        # tot = len(self.results) - no_judged
        # return tot,mrr / tot, float(count) / tot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', default="/home/longjing/zlz/Multi-CPR-main/retrieval/output/lawer_data/bert-base-chinese_cls.dev2.l2.tok5.out", type=str, help="search result")
    parser.add_argument('--qrel_path', default='../data/lawer_data/dev_case_to_lawer2.txt', type=str)
    parser.add_argument('--reverse', default=False, type=bool, help="reverse score during sorting, true for sparse and false for dense")
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--topk_list', default=[], type=list)
    parser.add_argument('--test_data',default="/home/longjing/zlz/Multi-CPR-main/data/lawer_mix/retrive_test_data.json",type=str)
    args = parser.parse_args()
    evaluation = evaluation(args.qrel_path, args.result_path, args.reverse)
    #tot,mrr, recall = evaluation.eval(args.topk)
    outlist=[]
    count, num, acc,outlist=evaluation.eval(args.topk,args.test_data)
    print (args.topk, count, num, acc)
    with open("./result/results5.json", "a", encoding="utf-8") as writer:
        writer.write(json.dumps({"data": outlist}, ensure_ascii=False))

