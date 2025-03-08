#include <iostream>
#include <vector>
#include <string>

using namespace std;

class ICapturer
{
public:
    virtual void stop() = 0;
    virtual bool start() = 0;
    virtual bool isCapturing() const = 0;
    virtual const string &last_error() const = 0;
    virtual vector<float> getAvailableSamples() const = 0;
    virtual ~ICapturer()
    {
        cout << "Destroyed " << endl;
    };

protected:
    ICapturer() = default; // Construtor protegido para evitar instanciação direta
};
