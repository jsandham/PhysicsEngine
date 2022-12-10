#include "../../include/core/World.h"
#include "../../include/core/Log.h"

#include "../../include/systems/AssetLoadingSystem.h"

#include "stb_image.h"
#include "stb_image_write.h"

using namespace PhysicsEngine;

//class ThreadPool {
//public:
//    void Start()
//    {
//        const uint32_t num_threads = std::thread::hardware_concurrency();
//        threads.resize(num_threads);
//        for (uint32_t i = 0; i < num_threads; i++) {
//            threads.at(i) = std::thread(ThreadLoop);
//        }
//    }
//
//    void QueueJob(const std::function<void()>& job)
//    {
//        {
//            std::unique_lock<std::mutex> lock(queue_mutex);
//            jobs.push(job);
//        }
//        mutex_condition.notify_one();
//    }
//    void Stop()
//    {
//        {
//            std::unique_lock<std::mutex> lock(queue_mutex);
//            should_terminate = true;
//        }
//        mutex_condition.notify_all();
//        for (std::thread& active_thread : threads) {
//            active_thread.join();
//        }
//        threads.clear();
//    }
//    void busy()
//    {
//        bool poolbusy;
//        {
//            std::unique_lock<std::mutex> lock(queue_mutex);
//            poolbusy = jobs.empty();
//        }
//        return poolbusy;
//    }
//
//private:
//    void ThreadLoop()
//    {
//        while (true) {
//            std::function<void()> job;
//            {
//                std::unique_lock<std::mutex> lock(queue_mutex);
//                mutex_condition.wait(lock, [this] {
//                    return !jobs.empty() || should_terminate;
//                    });
//                if (should_terminate) {
//                    return;
//                }
//                job = jobs.front();
//                jobs.pop();
//            }
//            job();
//        }
//    }
//
//    bool should_terminate = false;           // Tells threads to stop looking for jobs
//    std::mutex queue_mutex;                  // Prevents data races to the job queue
//    std::condition_variable mutex_condition; // Allows threads to wait on new jobs or termination 
//    std::vector<std::thread> threads;
//    std::queue<std::function<void()>> jobs;
//};




void AssetLoadingSystem::doWork(int slot)
{
	while (true)
	{
        //std::cout << "spin looping..." << std::endl;

        if (!mWorkerRequests[slot].mDone.load(std::memory_order_acquire))
        {
            std::cout << "started loading texture..." << mWorkerRequests[slot].mFilepath << " queue size: " << mTextureQueue.size() << std::endl;
            stbi_set_flip_vertically_on_load(true);

            int width, height, numChannels;
            unsigned char* raw = stbi_load(mWorkerRequests[slot].mFilepath.c_str(), &width, &height, &numChannels, 0);

            if (raw != NULL)
            {
                TextureFormat format;
                switch (numChannels)
                {
                case 1:
                    format = TextureFormat::Depth;
                    break;
                case 2:
                    format = TextureFormat::RG;
                    break;
                case 3:
                    format = TextureFormat::RGB;
                    break;
                case 4:
                    format = TextureFormat::RGBA;
                    break;
                default:
                    return;
                }

                mWorkerRequests[slot].mData.resize(width * height * numChannels);
                for (size_t j = 0; j < mWorkerRequests[slot].mData.size(); j++)
                {
                    mWorkerRequests[slot].mData[j] = raw[j];
                }
                mWorkerRequests[slot].mWidth = width;
                mWorkerRequests[slot].mHeight = height;
                mWorkerRequests[slot].mFormat = format;

                stbi_image_free(raw);

                std::cout << "finished loading texture..." << mWorkerRequests[slot].mFilepath << std::endl;

                mWorkerRequests[slot].mDone.store(true, std::memory_order_release);
            }
        }

		std::this_thread::sleep_for(std::chrono::milliseconds(16));
	}
}

AssetLoadingSystem::AssetLoadingSystem(World* world, const Id& id) : System(world, id)
{
    mNumThreads = std::thread::hardware_concurrency();
    
    //mWorkerRequests.resize(mNumThreads);
    //mWorkers.resize(mNumThreads);

    for (unsigned int i = 0; i < mNumThreads; i++)
    {
        mWorkerRequests[i].mIdle = true;
        mWorkerRequests[i].mDone.store(true, std::memory_order_release);
        mWorkers[i] = std::thread(&AssetLoadingSystem::doWork, this, (int)i);
    }
}

AssetLoadingSystem::AssetLoadingSystem(World* world, const Guid& guid, const Id& id) : System(world, guid, id)
{
    mNumThreads = std::thread::hardware_concurrency();
 
    //mWorkerRequests.resize(mNumThreads);
    //mWorkers.resize(mNumThreads);

    for (unsigned int i = 0; i < mNumThreads; i++)
    {
        mWorkerRequests[i].mIdle = true;
        mWorkerRequests[i].mDone.store(true, std::memory_order_release);
        mWorkers[i] = std::thread(&AssetLoadingSystem::doWork, this, (int)i);
    }
}

AssetLoadingSystem::~AssetLoadingSystem()
{
    for (unsigned int i = 0; i < mNumThreads; i++)
    {
        mWorkers[i].join();
    }
}

void AssetLoadingSystem::serialize(YAML::Node& out) const
{
    System::serialize(out);
}

void AssetLoadingSystem::deserialize(const YAML::Node& in)
{
    System::deserialize(in);
}

int AssetLoadingSystem::getType() const
{
    return PhysicsEngine::ASSETLOADINGSYSTEM_TYPE;
}

std::string AssetLoadingSystem::getObjectName() const
{
    return PhysicsEngine::ASSETLOADINGSYSTEM_NAME;
}

void AssetLoadingSystem::init(World* world)
{
    mWorld = world;
}

void AssetLoadingSystem::update(const Input& input, const Time& time)
{
    //std::cout << "Queue size: " << mTextureQueue.size() << std::endl;
    for (unsigned int i = 0; i < mNumThreads; i++)
    {
        if (mWorkerRequests[i].mIdle)
        {
            if (!mTextureQueue.empty())
            {
                std::string filepath = mTextureQueue.front();
                mTextureQueue.pop();

                mWorkerRequests[i].mFilepath = filepath;
                mWorkerRequests[i].mIdle = false;
                mWorkerRequests[i].mDone.store(false, std::memory_order_release);
            }
        }
    }

    for (unsigned int i = 0; i < mNumThreads; i++)
    {
        if (!mWorkerRequests[i].mIdle)
        {
            if (mWorkerRequests[i].mDone.load(std::memory_order_acquire))
            {
                Texture2D* texture = mWorld->createAsset<Texture2D>();
                texture->setRawTextureData(mWorkerRequests[i].mData, 
                                           mWorkerRequests[i].mWidth, 
                                           mWorkerRequests[i].mHeight, 
                                           mWorkerRequests[i].mFormat);

                mWorkerRequests[i].mIdle = true;
            }
        }
    }
}

void AssetLoadingSystem::loadTexture2DAsync(const std::string& filepath)
{
    mTextureQueue.push(filepath);
}
