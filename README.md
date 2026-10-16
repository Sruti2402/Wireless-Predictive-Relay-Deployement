# Predictive Wireless Relay Deployment for Disaster-Rescue Robotics

**Predict communication degradation. Deploy relay nodes before connectivity is lost.**

## Overview

In disaster-response environments, mobile robots may need to explore collapsed buildings, tunnels, or other hazardous areas where wireless communication with a base station is unreliable.

As an explorer robot moves farther from the base station or encounters obstacles and signal interference, communication quality can deteriorate. A conventional reactive approach deploys a relay only after connectivity has degraded. By then, valuable communication may already have been lost.

This project explores a **predictive, travel-time-aware relay deployment system** that uses wireless signal measurements and machine-learning predictions to estimate communication risk and determine when a relay should be dispatched.

The objective is to move from reactive relay deployment toward proactive communication maintenance.

## Problem Statement

How can a disaster-rescue robot maintain a communication path to its base station while exploring an environment where wireless connectivity changes over time?

A useful deployment decision must consider more than signal strength. It must also account for:

- Current and historical wireless signal measurements.
- Trends indicating deteriorating communication quality.
- The predicted time until a potential communication-risk event.
- Relay travel distance and movement speed.
- Relay setup time and network reconnection delay.
- A safety margin for uncertain operating conditions.

The system investigates whether predicting communication risk early enough can provide a feasible window for relay deployment.

## Proposed System

The project combines wireless-data analysis, machine-learning-based risk prediction, and a relay-dispatch evaluation framework.

### 1. Wireless signal monitoring

The dataset contains robot-position and wireless signal measurements, including:

- `robot_pos_x`, `robot_pos_y` — robot position.
- `UL_level`, `UR_level`, `LL_level`, `LR_level` — signal measurements from four directional regions.
- `C_level` — central signal measurement.
- Associated robot orientation, timing, and feedback fields.

These measurements provide information about the robot's movement and wireless environment.

### 2. Feature engineering

Raw measurements can be transformed into temporal features that help characterize communication behavior, such as:

- Recent signal trends.
- Changes in signal quality over time.
- Rolling statistics.
- Historical context preceding a potential risk event.

Temporal features should be computed without using future information that would be unavailable at prediction time.

### 3. Predictive risk estimation

Machine-learning models are evaluated to estimate the risk of a future communication-related event.

The project includes Random Forest and trend-based modeling experiments, along with out-of-fold prediction outputs.

The objective is to assess whether temporal patterns provide useful early-warning information beyond a simpler baseline.

**Important:** A model's prediction is an estimate of risk, not a guarantee that communication will fail. The chosen signal thresholds must be validated against an appropriate communication-failure definition before the system is used in a real rescue mission.

### 4. Travel-time-aware relay dispatch

Once a warning is available, the system evaluates whether a relay can be deployed before a predicted risk event.

The basic deployment-time calculation is:

\[
T_{\text{required}} =
\frac{D_{\text{relay}}}{V_{\text{relay}}}
+ T_{\text{setup}}
+ T_{\text{reconnect}}
+ T_{\text{safety}}
\]

Where:

- \(D_{\text{relay}}\): relay travel distance in metres.
- \(V_{\text{relay}}\): relay speed in metres per second.
- \(T_{\text{setup}}\): time required to position and configure the relay.
- \(T_{\text{reconnect}}\): estimated time to restore the communication path.
- \(T_{\text{safety}}\): additional safety allowance.

Let \(T_{\text{warning}}\) denote the predicted warning lead time.

A relay deployment is considered temporally feasible when:

\[
T_{\text{warning}} \geq T_{\text{required}}
\]

The remaining time margin is:

\[
T_{\text{margin}} =
T_{\text{warning}} - T_{\text{required}}
\]

A positive margin indicates that the modeled deployment can finish before the predicted event, under the assumed scenario conditions.

This is a feasibility estimate rather than a guarantee of successful deployment. Actual performance also depends on relay placement, network topology, terrain, and the accuracy of the warning.

## Relay Deployment Scenarios

The evaluation framework explores combinations of:

| Parameter | Example values |
|---|---|
| Relay travel distance | 5 m, 10 m, 15 m |
| Relay speed | 0.1 m/s, 0.2 m/s, 0.4 m/s |
| Setup time | 2 s, 5 s |
| Reconnection delay | 1 s, 3 s |
| Safety margin | 0 s, 2 s |

These are simulation scenario inputs, not claims about the performance of a physical robot.

The scenario framework helps investigate how deployment feasibility changes with relay mobility, setup overhead, and available warning time.

## Dashboard

The project includes a dashboard for presenting relay deployment analysis and experimental results.

The dashboard is intended to make the main operational question easier to understand:

**Should a relay be deployed now, and is there enough predicted warning time to deploy it before communication becomes unreliable?**

Depending on the configured analysis, the dashboard can present risk predictions, deployment scenarios, warning margins, and connectivity evaluation results.

Dashboard outputs should be interpreted in the context of the dataset and simulation assumptions used to generate them.

## Dataset

The project uses recorded indoor mobile-robot wireless measurements stored in `.datalog` files, including:

- `Dataset1.datalog`
- `Dataset2.datalog`
- `Dataset3.datalog`
- `Dataset4.datalog`
- `Dataset5.datalog`
- `Dataset6.datalog`
- `Dataset7.datalog`

The recordings include robot motion and multiple wireless signal measurements.

The data is processed into model-ready datasets and prediction files for training, validation, and relay-dispatch evaluation.

Large processed CSV files may be excluded from the Git repository because of GitHub file-size limits. They can be regenerated from the source recordings and processing scripts where the required inputs are available.

## Evaluation Approach

The project investigates predictive performance and operational usefulness separately.

### Predictive evaluation

Relevant metrics include:

- **Precision:** How many predicted risk events correspond to actual labeled events?
- **Recall:** How many actual labeled events are detected?
- **F1-score:** The balance between precision and recall.
- **Average precision:** How well the model ranks positive events across different thresholds.
- **Cross-recording validation:** Testing on recordings held out from model training to assess generalization.

### Operational evaluation

Relevant deployment metrics include:

- Warning coverage across evaluated risk events.
- Number of false alerts.
- Percentage of events with sufficient time for relay deployment.
- Sensitivity to relay distance, speed, setup time, and reconnection delay.
- Comparison between predictive and reactive dispatch policies.

Predictive accuracy alone is insufficient: an accurate warning that arrives too late to deploy a relay may have little operational value.

## Repository Structure

The repository is organized around source code, recorded data, processed outputs, and dashboard components. The exact contents may evolve as experiments are added.

```text
predictive-relay-deployment/
├── src/                    # Data processing, modeling and evaluation scripts
├── data/
│   ├── raw/                 # Original wireless recordings
│   └── processed/           # Processed datasets and predictions
├── dashboard/               # Dashboard and analysis visualizations
├── relay_dashboard/          # Additional dashboard application files
├── .gitignore
├── requirements.txt         # Python dependencies, if provided
└── README.md
```

Large datasets, generated prediction files, virtual environments, and temporary experiment outputs may be excluded from version control. Consult `.gitignore` and the available scripts for the current configuration.

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the project dependencies.
- Git.
- The required dataset recordings.
- The dependencies listed in `requirements.txt`, if available.

### 1. Clone the repository

```bash
git clone https://github.com/Sruti2402/Wireless-Predictive-Relay-Deployement.git
cd Wireless-Predictive-Relay-Deployement
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If the repository contains `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the required wireless recordings in the location expected by the data-processing scripts. Check each script's configuration before running it.

### 5. Run the experiments

Execute the relevant scripts in their intended order, beginning with data preparation, followed by feature generation, model evaluation, and relay-dispatch simulation.

The precise commands depend on the scripts included in the repository. Refer to their entry points and configuration before execution.

### 6. Launch the dashboard

Follow the instructions associated with the dashboard application in the repository. The launch command depends on whether the dashboard is served as a static page or through a separate application framework.

## Limitations and Future Work

This project is an experimental research framework, not yet a validated disaster-response deployment system.

Future work includes:

- Validating signal-risk labels against actual communication failures.
- Testing temporal generalization on independent recordings.
- Calibrating predicted warning times and risk probabilities.
- Comparing predictive dispatch against reactive baselines under identical conditions.
- Modeling relay placement, connectivity topology, and multi-hop communication paths.
- Evaluating uncertain travel times and failed relay deployments.
- Integrating live wireless telemetry from an explorer robot.
- Testing the complete system with physical robots in controlled environments.

## Intended Application

The long-term application is disaster-rescue robotics, where maintaining a reliable communication path between an explorer robot and a remote base station can support safer exploration and more effective mission coordination.

The system aims to help answer a practical question:

**Can we recognize deteriorating communication early enough to deploy a relay before the explorer loses its communication path to the base station?**

## License

Add the license appropriate to the project's intended use before distributing or reusing the code.

## Acknowledgments

The project builds on recorded mobile-robot wireless measurements and explores predictive modeling and travel-time-aware relay deployment for resilient robotic communication.
