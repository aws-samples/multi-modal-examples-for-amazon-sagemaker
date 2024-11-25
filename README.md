![Banner](/media/banner-large.svg)

# Amazon SageMaker Multi-Modal Samples

This repository contains pre-built examples to help customers get started with the Amazon SageMaker and Multi-Modal Large Language Models (MLLMs). 


## Multi Modal Labs

- [01 Video Content Highlight Reel Generation with Qwen2-VL](01-video_content_reel_generator-qwen2_vl)



> ⚠️ **Note :** All Labs are built and tested on SageMaker Studio JupyterLab IDE or SageMaker Notebooks. Please feel free to create to cut us a ticket if you experience any issues running this inside other IDEs.


## Get Started with Amazon SageMaker

1. Navigate to [Amazon SageMaker](https://aws.amazon.com/sagemaker/) from AWS Console

![Navigate to Amazon SageMaker](/media/search-for-sagemaker.png)

2. To get started using these notebooks - Select your Domain and UserProfile from the drop-down. If you don't have a [SageMaker Studio Domain](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) - refer to [Create a new Studio Domain](#create-a-new-sagemaker-studio-domain)

![Launch a UserProfile](/media/launch-a-userprofile.png)


## Create a new SageMaker Studio Domain

To create a new SageMaker Studio Domain follow [Option 1: Set up for single user (Quick setup)] or [Option 2: Set up for organizations]

### [Option 1: Set up for single user (Quick setup)](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html)

The Set up for single users (quick setup) procedure gets you set up with default settings. Use this option if you want to get started with SageMaker quickly and you do not intend to customize your settings at this time. The default settings include granting access to the common SageMaker services for individual users to get started. For example, Amazon SageMaker Studio and Amazon SageMaker Canvas.

To quickly launch a new domain and get started with JupyterLab IDE,
![Create a Domain](/media/create-domain.png)


And setup a new domain. That's all!

![Setup Quick Domain](/media/quick-domain-setup.png)


### [Option 2: Set up for organizations](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-custom.html)

The Set up for organizations (custom setup) guides you through an advanced setup for your Amazon SageMaker domain. This option provides information and recommendations to help you understand and control all aspects of the account configuration, including permissions, integrations, and encryption. Use this option if you want to set up a custom domain.

To get started use the following infrastructure as code templates,
- [SageMaker inside VPC Cloudformation](/sagemaker-cloudformation/sagemaker-in-vpc.yaml)
- [SageMaker inside VPC Terrafrom](/sagemaker-cloudformation/sagemaker-in-vpc.tf)


## Contributing

We welcome community contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
