#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>   //陣列
#include <cstdlib>
using namespace std;

vector<string> _csv(string s);

int main()
{
    ifstream inFile("mnist_test.csv", ios::in);
    if (!inFile)
    {
        cout << "開啟檔案失敗！" << endl;
        exit(1);
    }   
    string line;  
    while (getline(inFile, line))
    {
        cout << "org=" << line << endl;   
        //========================
        vector<string> a = _csv(line);
        cout << "size=" << a.size() << endl;
        for (int ii = 0; ii < a.size(); ii++)
        {
            cout << a[ii] << ",";
        }
        cout << endl;
        //========================
    }
}


vector<string> _csv(string s)
{
    vector<string> arr;
    istringstream delim(s);
    string token;
    int c = 0;
    while (getline(delim, token, ','))        
    {
        arr.push_back(token);                
        c++;                                           
    }
    return  arr;
}
