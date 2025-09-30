#set page(
  paper: "us-letter",
  margin: (x:1.8cm, y:1.6cm),
  header: context if here().page()==1{""} else {align(right)[TOC and LSTM Based Resource Scheduling]},
  numbering: "1",
  // columns: 2
)
#set text(
  font: "New Computer Modern",
  size: 12pt
)
#set par(
  justify: true,
  leading: 0.52em
)
#set heading(
  numbering:"1.1" 
)
#place(
  top + center,
  float: true,
  scope: "parent",
  clearance: 2em,
)[
  #align(center, text(17pt)[
    *TOC and LSTM Based Resource Scheduling*
  ])
  #grid(
    columns: (1fr, 1fr,1fr),
    align(center)[
      Nipun S Nair \
      VIT Vellore \
    ],
    align(center)[
      Anamika \
      VIT Vellore \
    ],
    align(center)[
      Vishnu P K  \ 
      VIT Vellore \
    ],
  )

  #align(center)[
    #set par(justify: false)
    *Abstract* \
    #lorem(80)
  ]
]

= Literature Review
Chang et al. @Chang2017ApplyingTO applied a TOC-Based approach to address memory allocation in cloud storage, which uses market information to build a rolling forecast. This demonstrates that TOC principles can be extended to resource scheduling problems in dynamic environments.

= Methodology
This section describes the procedures, tools, and methods used to conduct the research.

== Training LSTM Model to predict bottlenecks
=== Synthetic Data Generation
We simulate a realistic time-series dataset representing server resource usage over time. The goal is to generate data that mimics real-world system behavior to train an LSTM model for bottleneck prediction.

#figure(
  caption: [Simulation Parameters],
  kind: table, 
  [
    #table(
      columns: (auto,auto),
      align: left,
      stroke: 0.4pt,
      [*Parameter*], [*Description*],
      [NUM_SERVERS], [Number of simulated servers],
      [SIM_TIME], [Total simulated duration (in time units)],
      [TASK_INTERVAL], [Mean inter-arrival time for new tasks],
      [CPU_CAPACITY], [Max CPU Usage (100%)],
      [NET_CAPACITY], [Max network bandwidth (100%)],
    ) 
  ]
)<simulation_parameters>

==== Poisson-Based Task Arrival
Task arrival times are based on a Poisson process, simulated using the exponential distribution:

$ "Interarrival Time" ~ "Exponential"(lambda = 1 / "TASK_INTERVAL") $

This introduces realistic irregularity in task arrivals—mimicking user requests, job submissions, or packet arrivals.

==== Time-Varying Resource Usage Patterns
We simulate periodic system load using sinusoidal functions to represent diurnal load patterns, cyclic CPU spikes, and network traffic fluctuations.

*CPU Usage*
$ "cpu_base"(t) = 50 + 40 * sin((2pi t) / "PERIOD") $

*Network In/Out*
$ "net_base"(t, s) = 30 + 25 * sin((2pi (t + s * 10)) / "PERIOD") $

Where:
- $t$: current timestamp
- $s$: server index (adds phase shift between servers)
- `PERIOD`: defines workload cycle length (e.g. peaks every 100 time units)

==== Gaussian Noise for Realism
We inject Gaussian noise to simulate sensor or monitoring variability. This produces "wobble" on top of clean trends, similar to real monitoring data.

$ "cpu" = "clip"("cpu_base" + cal(N)(0, 10), 0, 100) $

$ "net_in", "net_out" = "clip"("net_base" + cal(N)(0, 5), 0, 100) $

==== Scheduled Bottlenecks
At specific intervals (e.g., for $t = 250$ to $260$ on Server 1), we introduce high-load conditions to ensure predictable bottlenecks for model learning and testing.
- CPU, Net In, and Net Out are all set to $>= 90%$.

==== Bottleneck Label Definition
A binary bottleneck label is assigned based on an 80% utilization threshold:

$ "bottleneck" = cases(
  1, "if CPU" >= 80 " or Net In/Out" >= 80,
  0, "otherwise"
) $

#figure(
  image("./TrainingLSTM/SyntheticData.png"),
  caption: "Synthetic Data - Compute and Network Resources"
)<syntheticData>

=== Training Model
To identify potential bottlenecks, we trained a Long-Short-Term Memory neural network, as it is well suited for learning patterns and dependencies in time series data.

To capture temporal patterns, the continuous time series data for each server was transformed into overlapping sequences of windows of size 20 timesteps, with each window serving as an input sequence and the following timestep's bottleneck status as the label (1/0). This information allows the model to learn the patterns leading up to a bottleneck.

The dataset was then split into training and test sets and a MinMax scaler was fit only on the training data and applied to both the sets. Finally we trained a recurrent neural network with an LSTM layer followed by dense layers optimized using binary cross-entropy loss, to classify whether a bottleneck would occur in the next timestep, given the previous 20 timesteps.

=== Evaluation
The model was evaluated on a test set of 293 samples, achieving an overall accuracy of 89%. The detailed performance metrics from the classification report are presented below.
#figure(
  caption: [Classification Report],
  kind: table,
  [
    #table(
      columns: 5,
      align: (center, center, center, center, center),
      stroke: 0.4pt,
      [], [*Precision*], [*Recall*], [*F1-Score*], [*Support*],
      [*No Bottleneck (0)*], [0.94], [0.91], [0.93], [235],
      [*Bottleneck (1)*], [0.69], [0.78], [0.73], [58],
      [], [], [], [], [],
      [*Accuracy*], [], [], [0.89], [293],
      [*Macro Avg*], [0.82], [0.85], [0.83], [293],
      [*Weighted Avg*], [0.89], [0.89], [0.89], [293],
    ) 
  ]
)<classification_report>

#figure(
  caption: [Confusion matrix of model predictions.],
  kind: table,
   [
    #table(
      columns: (auto, auto, auto),
      align: center,
      stroke: 0.4pt,
      [], [*Predicted: No Bottleneck*], [*Predicted: Bottleneck*],
      [*Actual: No Bottleneck*], [215], [20],
      [*Actual: Bottleneck*], [13], [45],
    )
  ],
) <confusion_matrix>

== Algorithm 1 (Base) - Round Robin Simulation
The core principle of Round Robin is to achieve fairness and prevent starvation by distributing tasks in a simple, cyclical sequence by employing a stateless, turn-based approach. It maintains a pointer to the last server that received a task and assigns the next incoming task to the subsequent server in the sequence.

Our *resource aware* algorithm performs two crucial checks to make sure there will be no queue overloads or system failure due to capacity overloads.
1.  *Queue Capacity Check:* The server's task queue must not be full.
2.  *Resource Availability Check:* The server must have sufficient free CPU and network capacity to begin processing the specific task at that moment.

If the designated server fails these checks, the algorithm continues its cyclical search until an adequate server is found. If no server in the system can accept the task, it is rejected.

=== Algorithm Steps

Let $bb(S)$ be the set of $N$ servers, indexed as $bb(S) = \{s_0, s_1, ..., s_(N-1)\}$. Let $I_"last"$ be the index of the server that received the previous task. The scheduling process for each new incoming task, $J_"new"$, is executed as follows:

+ *Initialization:* The algorithm identifies the starting index for its search, $I_"start"$, based on the last server to receive a task:
  $
    I_"start" = (I_"last" + 1) mod N
  $

+ *Cyclical Search:* The algorithm iterates through all active servers $s_k$ in a circular order for $N$ steps, starting from the server at index $I_"start"$.

+ *Eligibility Check:* For each candidate server $s_k$, the algorithm determines its eligibility by evaluating two boolean conditions. The task, $J_"new"$, arrives with a set of resource demands $D = \{D_"cpu", D_"net_in", D_"net_out"\}$. The state of server $s_k$ at time $t$ is defined by its queue length $Q_(s_k)(t)$ and its available capacities for each resource $r in {"cpu", "net_in", "net_out"}$, denoted $C_"avail"^r (s_k,t)$.

  1.  *The Queue Capacity Condition*, $"Accept"_Q(s_k)$, must be true:
      $
        "Accept"_Q(s_k) = [ Q_(s_k)(t) < Q_"max" ]
      $
      where $[.]$ is the Iverson bracket.

  2.  *The Resource Availability Condition*, $"Accept"_R(s_k, J_"new")$, must also be true:
      $
        "Accept"_R(s_k, J_"new") = [D_"cpu" <= C_"avail"^"cpu" (s_k,t)] and [D_"net_in" <= C_"avail"^"net_in" (s_k,t)] and [D_"net_out" <= C_"avail"^"net_out" (s_k, t)]
      $

+ *Assignment or Rejection:* The first server $s_k$ in the sequence for which both conditions are met is selected as the target server, $bb(S)_"target"$.
  $
    bb(S)_"target" = s_k quad "where" quad "Accept"_Q(s_k) and "Accept"_R(s_k, J_"new")
  $
  Upon successful assignment, the index $I_"last"$ is updated to $k$, and the search terminates. If the algorithm completes a full cycle and no server satisfies both conditions, the task $J_"new"$ is rejected, constituting an SLA violation.

=== Architecture
#figure(
  image("/TrainingLSTM/RR_architecture.png"),
  caption: "Architecture of Round Robin Scheduler"
)<RRarchitecture>

== Algorithm 2 - Theory of Constraints Simulation
  This algorithm provides a practical implementation of TOC's *Drum-Buffer-Rope (DBR)* methodology for a parallel, dynamic server environment.

- *The Drum:* The system's identified constraint, whose processing rate dictates the optimal rate at which new work should be introduced.
- *The Buffer:* A small, managed queue of tasks placed before each resource. The buffer in front of the constraint is the most critical, as it must ensure the constraint is never idle due to a lack of work.
- *The Rope:* A signaling mechanism that links the constraint's buffer back to the system's entry point. It authorizes the release of new work into the system only when the constraint has the capacity to process it, thereby synchronizing the entire system to the pace of its slowest part.

The algorithm is composed of four primary components that map to the Five Focusing Steps of TOC: Constraint Identification, Flow Control (Dispatcher), Task Assignment, and System Scaling.

===  Algorithm Steps

Let $bb(S)$ be the set of all servers. At any time $t$, the set of active servers is denoted by $bb(S)_"active"(t) subset.eq bb(S)$. Each server $s in bb(S)$ has a maximum CPU capacity $C_(s,"cpu")$ and network capacity $C_(s,"net")$.
==== Constraint Identification (Identify)

The first step is to dynamically and continuously identify the system's primary constraint. Instantaneous resource utilization is often volatile; therefore, a smoothing function is required to identify the most persistently loaded resource. We employ an *Exponentially Weighted Moving Average (EWMA)* for this purpose.

Let $U_(s,r)(t)$ be the instantaneous utilization of a resource $r in {"cpu", "net"}$ on a server $s$ at time $t$. The utilization is a normalized value where $0 <= U <= 1$.

The smoothed utilization, $#overline("U")_(s,r)(t)$, is calculated recursively:
$
  #overline("U")_(s,r)(t) = (alpha dot U_(s,r)(t)) + ((1 - alpha) dot #overline("U")_(s,r)(t - Delta t))
$
Where:
- $alpha$ is the smoothing factor ($0 < alpha < 1$). A lower $alpha$ results in a smoother, less volatile trendline.
- $Delta t$ is the time interval between measurements.

The system constraint at time $t$, denoted $C(t)$, is the specific resource $(s, r)$ with the highest smoothed utilization across all active servers.
$
  C(t) = op("argmax")_(s in S_"active"(t), r in {"cpu", "net"}) { #overline("U")_(s,r)(t) }
$
This identification process runs at a fixed interval, `CONSTRAINT_CHECK_INTERVAL`, to adapt to changing system loads.

==== Flow Control via Dispatcher (Exploit & Subordinate)

The core of the DBR implementation is a centralized *Dispatcher* that acts as the "Rope." It manages a central priority queue of incoming tasks, $B_"central"$, and only releases work into the system based on the state of the identified constraint, $C(t)$.

Let $C_s(t)$ be the server component of the constraint $C(t)$. Let $Q_s(t)$ be the length of the local buffer (queue) of server $s$ at time $t$, and let $Q_"max"$ be the maximum configured size of this buffer.

The *Rope Condition*, $"Release"(t)$, is a boolean function that determines if a new task should be released from $B_"central"$:
$
  "Release"(t) = [Q_(C_s(t))(t) < Q_"max"]
$
This condition ensures that a new task is only introduced into the system when the constraint's buffer has capacity. This prevents the accumulation of excessive Work-in-Process (WIP) and paces the entire system to its bottleneck. If $"Release"(t)$ is false, no tasks are dispatched, and they remain in the managed, prioritized central buffer.

==== Task Assignment (Subordinate)

When the Rope Condition $"Release"(t)$ is met, the highest-priority task, $J_"next"$, is selected from the central buffer:
$
  J_"next" = op("argmin")_(j in B_"central") { P(j) }
$
where $P(j)$ is the priority value of task $j$ (lower is higher).

This task must then be assigned to an active server. This is a subordinate decision, designed to efficiently utilize non-constraint resources without disturbing the system's flow. The target server, $bb(S)_"target"$, is selected from the set of available servers, $bb(S)_"avail" (t)$, by finding the server with the smallest local buffer.

The set of available servers is defined as:
$
  bb(S)_"avail" (t) = { s in bb(S)_"active" (t) | Q_s (t) < Q_"max" }
$
The target server is then chosen by:
$
  bb(S)_"target" = op("argmin")_(s in bb(S)_"avail" (t)) { Q_s (t) }
$
This ensures the released task is routed to the most idle part of the system, minimizing its local wait time and keeping non-constraint resources productive.

==== System Scaling (Elevate)

The final component is the autoscaler, which implements the "Elevate the Constraint" step. It modifies the size of the active server set, $abs(bb(S)_"active" (t))$.

Let $theta_"up"$ be the scale-up threshold and $theta_"down"$ be the scale-down threshold. Let $N(t) = abs(bb(S)_"active" (t))$ be the number of active servers.

*Scale-Up Condition:* The decision to scale up is based solely on the status of the constraint. If the smoothed utilization of the constraint resource exceeds the threshold, a new server is activated.
$
  "ScaleUp"(t) = [#overline("U")_(C(t))(t) > theta_"up"] and [N(t) < N_"max"]
$
This ensures that capacity is added precisely where it is needed to relieve the system's bottleneck.

*Scale-Down Condition:* The decision to scale down is based on overall system idleness. Let $U.bar_"sys"(t)$ be the average CPU utilization across all active servers. To prevent premature scaling during the initial warm-up phase, a time condition $T_"warmup"$ is included.
$
  "ScaleDown"(t) = [t > T_"warmup"] and [#overline("U")_"sys"(t) < theta_"down"] and [N(t) > N_"min"]
$
This allows the system to conserve resources when the overall demand is low, without being triggered by the intentionally low utilization of non-constraint servers during periods of high load.

=== Architecture
#figure(
  image("/TrainingLSTM/TOC_architecture.png"),
  caption: "Architecture of Theory of Constraints Scheduler"
)<TOCarchitecture>
== Algorithm 3 - Theory of Constraints with LSTM Bottleneck Prediction

The final algorithm represents the synthesis of the preceding methodologies, integrating the predictive capabilities of the trained LSTM model into the TOC framework. This creates a proactive, intelligent scheduling system that anticipates bottlenecks rather than merely reacting to them.


=== Algorithm Steps

The algorithm's components are refactored to incorporate the predictive model. The *ConstraintDetector* is now AI-driven, and its output directly influences the *Dispatcher* and the *Autoscaler*.

==== Predictive Constraint Identification

The reactive, utilization-based constraint identification is replaced by a predictive process that leverages the trained LSTM model. This component's goal is to forecast which server is most likely to become a bottleneck in the near future.

Let $s_i$ be an active server. At time $t$, its state over the last $W$ timesteps (the `WINDOW_SIZE`) is represented by a feature matrix $X_(s_i)(t) in RR^(W times F)$, where $F$ is the number of features (CPU, Queue Length, Network In, Network Out).

1.  *Feature Scaling:* The raw feature matrix $X_(s_i)(t)$ is normalized using the pre-trained scaler function, $g_"scaler"$:
    $
      hat(X)_s_i (t) = g_"scaler" (X_(s_i)(t))
    $

2.  *Inference:* The normalized feature matrix is fed into the trained LSTM model, $f_"LSTM"$, which outputs a bottleneck probability score, $P_"bottleneck"$.
    $
      P_"bottleneck"(s_i, t) = f_"LSTM"(hat(X)_(s_i)(t))
    $

The system's predicted constraint at time $t$, $C_p (t)$, is the server with the highest probability score.
  $
    C_p (t) = op("argmax")(s_i in bb(S)_("active")(t)) { P_"bottleneck"(s_i, t)}
  $

==== Task Dispatching via Constraint Avoidance

The *Dispatcher* logic is inverted from the DBR model. Instead of subordinating to the constraint's pace, it actively works around the predicted constraint to prevent pile-ups.

For an incoming task $J_"new"$, the target server, $bb(S)_"target"$, is selected as follows:

1.  *Define the Candidate Pool:* First, a set of eligible, non-constraint servers, $bb(S)_("eligible")(t)$, is created.
    $
      bb(S)_"eligible"(t) = {s in bb(S)_"active"(t) | s != C_p (t) and Q_s (t) < Q_"max" }
    $

2.  *Primary Assignment:* If the eligible pool is not empty, the task is assigned to the server with the minimum queue length within that pool.
    $
      bb(S)_"target" = op("argmin")_(s in bb(S)_"eligible" (t)) { Q_s (t) }
    $

3.  *Fallback Assignment:* If, and only if, the eligible pool is empty (meaning all non-constraint servers are at maximum capacity), the system will attempt to assign the task to the constraint server, $C_p (t)$, provided it has queue space. This prevents a total system stall when under extreme load.

==== Hybrid System Scaling

The autoscaler is enhanced with a hybrid proactive/reactive policy to make it both intelligent and resilient.

Let $P_c (t)$ be the bottleneck score of the predicted constraint server, $C_p (t)$. Let $U_c (t)$ be the *current* maximum resource utilization (CPU or Network) of that same server. The scale-up decision is triggered by either a proactive or a reactive condition.

1.  *Proactive Trigger (AI-Driven):* Scale up if the AI's confidence in an impending bottleneck is high.
    $
      "Trigger"_"proactive" = [ P_c (t) > theta_"up_prob" ]
    $

2.  *Reactive Trigger (Failsafe):* Scale up if the predicted constraint is *already* in a danger zone, even if the AI's score is not high. This protects against sudden, unpredicted load spikes.
    $
      "Trigger"_"reactive" = [ U_c (t) > theta_"danger" ]
    $

The final scale-up condition is a logical OR of these two triggers:
$
  "ScaleUp"(t) = ("Trigger"_"proactive" or "Trigger"_"reactive") and [N(t) < N_"max"]
$
where $N(t)$ denotes the number of active servers at time $t$, and $N_"max"$ is the maximum configured server limit for the system.

The scale-down logic remains unchanged from Algorithm 2, providing stability by only removing resources when the entire system is demonstrably idle.

=== Architecture
#figure(
  image("/TrainingLSTM/TOC_AI_architecture.png"),
  caption: "Architecture of AI driven Theory of Constraints Scheduler"
)<TOCAIarchitecture>


= Results and Analysis

This section presents a comparative performance analysis of the three implemented scheduling algorithms. The goal is to measure throughput, quality of service, and resource utilisation in order to assess how well they manage a dynamic workload.

== Experimental Setup

A discrete-event simulation environment created in Python with the `SimPy` library was used to assess the performance of the three scheduling algorithms.  @tbl-sim-params contains a detailed description of the main parameters controlling the workload, scheduler policies, and simulation environment.

The workload was generated to mimic realistic, time-varying demand. Task arrivals follow a Poisson process, while the resource demands for each task are based on a cyclical, sinusoidal pattern combined with Gaussian noise to simulate diurnal variations and random fluctuations.

#figure(
  table(
    columns: (0.5fr, 1fr),
    align: (left, left),
    table.header(
      [*Category*], [*Parameter & Value*]
    ),
    table.hline(),
    
    [System], [
      *Simulation Duration:* 500 virtual seconds \
      *Initial Server Count:* 3 \
      *Min / Max Servers:* 2 / 10 \
      *Server CPU Capacity:* 100 units \
      *Server Network Capacity:* 100 units
    ],

    [Workload], [
      *Mean Task Arrival Interval:* 1.0 seconds (Poisson) \
      *Task CPU Demand:* $ C_text(t) = 70 + 40sin((2pi t)/100) + cal(N)(0, 10) $ \
      *Task Network Demand:* $ N_text(t) = 50 + 25sin((2pi t)/100) + cal(N)(0, 5) $
    ],

    [TOC], [
      *Server Buffer Size (`MAX_QUEUE_LEN`):* 5 tasks \
      *Dispatcher Buffer Size (`MAX_CENTRAL_BUFFER_LEN`):* 50 tasks
    ],

    [Autoscaling], [
      *Scaling Check Interval:* 15 seconds \
      *Reactive Scale-Up Threshold:* 75% of constraint utilization \
      *Predictive Scale-Up Threshold:* 60% of bottleneck probability \
      *Scale-Down Threshold:* 40% of average system utilization
    ],
  ),
  caption: [Simulation Parameters for the `SCALABLE_BALANCED` Configuration.],
) <tbl-sim-params>


== Performance Metrics

Five Key Performance Indicators (KPIs) form the basis of the evaluation:
- *Completion Rate:* The system throughput is measured by the proportion of all generated tasks that were successfully processed..
- *SLA Violation Rate:* System reliability is measured by the proportion of all generated tasks that were rejected because of system overload.
- *Average CPU & Network Utilization:* The average resource usage for all _active_ servers, which gauges resource efficiency.
- *Average Turnaround Time:* The mean amount of time, measured in latency, between the arrival of a task and its completion.

== Comparative Performance Analysis

The aggregated results from the simulations are summarized in @tbl-results-summary. The data reveals a clear performance hierarchy among the three algorithms, which is analyzed in the subsequent sections.

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center),
    table.header(
      [*Performance Metric*], [*Round Robin (RR)*], [*TOC (Reactive)*], [*AI+TOC (Predictive)*]
    ),
    table.hline(),
    "Completion Rate (%)", "60.66", "92.15", "94.11",
    "SLA Violation Rate (%)", "37.68", "3.21", "0.00",
    "Avg. CPU Utilization (%)", "65.11", "81.61", "79.11",
    "Avg. Network Utilization (%)", "48.49", "59.27", "57.39",
    "Avg. Turnaround Time (s)", "11.62", "52.84", "37.64",
  ),
  caption: [Comparative Performance of Scheduling Algorithms.],
) <tbl-results-summary>

=== Throughput and System Stability

The most significant finding is the dramatic improvement in both throughput and stability offered by the TOC-based schedulers. As shown in @fig:throughput-sla, the baseline RR scheduler struggled under the dynamic workload, successfully completing only *60.66%* of tasks and suffering a very high SLA Violation Rate of *37.68%*. This is characteristic of a "push" system, where uncontrolled task assignment leads to cascading queue overloads and high rejection rates.

In contrast, the reactive TOC scheduler, by implementing a "pull" system via the Drum-Buffer-Rope logic, achieved a remarkable increase in stability. It successfully completed *92.15%* of tasks while reducing the SLA Violation Rate to just *3.21%*.

With a nearly flawless completion rate of *94.11%* and a SLA violation rate of *0.00%*, the Hybrid AI+TOC scheduler is the best in its class.  This illustrates the effectiveness of proactive scaling; the AI-driven system was able to "elevate" its capacity before the central buffer became saturated, thereby completely removing task rejections, by using the LSTM to anticipate approaching bottlenecks.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    image("/TrainingLSTM/Completion Rate.png", width: 100%, fit: "contain"),
    image("/TrainingLSTM/SLA Rate.png", width: 100%, fit: "contain"),
  ),
  caption: [Comparison of Completion Rate (left) and SLA Violation Rate (right). The AI+TOC model's proactive scaling eliminated SLA violations],
) <fig:throughput-sla>

=== Resource Utilization

The resource utilization metrics in @fig:utilization provide insight into the operational efficiency of each scheduler. Contrary to expectations, the less effective RR scheduler uses less CPU (*65.11%*) and network (*48.49%*) on average than the TOC models.  This decreased utilisation is a symptom of system chaos rather than efficiency; the scheduler wastes a lot of time rejecting tasks that it is unable to place, which results in idle resources amid the overload.

The TOC and AI+TOC schedulers maintain a higher and more productive level of resource utilization (Avg. CPUs of *79.11%* and *81.61%*, respectively). This demonstrates the TOC's fundamental idea, that the system can keep its resources working on value-adding tasks (processing tasks) rather than wasting them or thrashing them by safeguarding the constraint and maintaining a smooth workflow.

#figure(
  image("/TrainingLSTM/Utilization.png", width: 100%, fit: "contain"),
  caption: [Resource Utilization Analysis Across Schedulers. The higher utilization of the TOC models indicates greater processing efficiency],
) <fig:utilization>

=== System Latency

An analysis of task turnaround time distribution, visualized in the box plot in @fig:turnaround, reveals the fundamental performance trade-offs of each scheduling philosophy.

The Round Robin (RR) scheduler exhibits the lowest median latency, though this result is misleading. Its speed is a direct consequence of a high SLA violation rate (*37.68%*); it achieves low latency by processing only the tasks that arrive during periods of low load while rejecting a large portion of the workload.

In contrast, the reactive TOC scheduler prioritizes stability, resulting in a significantly higher median latency due to tasks waiting in a central buffer (a high Work-in-Process state). The plot's wide distribution and numerous extreme outliers show that while the system avoids rejections, it suffers from unpredictable, long delays during congestion.

The Hybrid AI+TOC scheduler provides the most balanced approach. It reduces the median latency compared to the reactive TOC model and shows a tighter interquartile range (IQR), indicating greater predictability. This improvement is a direct result of proactive scaling mitigating system congestion. While the presence of some high-end outliers indicates the AI is not immune to prediction errors, the analysis confirms a classic trade-off: the AI+TOC model successfully improves both latency and consistency over the reactive model, sacrificing the illusory speed of RR for genuine system stability and throughput.

#figure(
  image("/TrainingLSTM/Latency.png", width: 80%),
  caption: [Comparison of Average Task Turnaround Time. The higher latency of the TOC models is a deliberate trade-off for system stability, which the AI model helps to mitigate],
) <fig:turnaround>

= Conclusion

This paper developed and evaluated a proactive, hybrid scheduling system for parallel server environments by integrating the TOC with a predictive LSTM model. The research successfully demonstrates that a constraint-aware, predictive scheduling philosophy yields significant and measurable improvements in system throughput, stability, and efficiency over traditional methods.

The baseline RR scheduler, representing a stateless "push" system, proved incapable of managing the dynamic workload, resulting in significant task rejection and low throughput. This highlights the inherent limitations of simple load distribution strategies under variable load conditions.

The introduction of a reactive TOC scheduler marked a significant paradigm shift. By implementing a "pull" system via the Drum-Buffer-Rope methodology, the scheduler achieved a dramatic improvement in system stability and overall task completion. This performance gain, however, came at the cost of increased task latency, a deliberate architectural trade-off inherent to the DBR model for maintaining system control.

The final Hybrid AI+TOC scheduler demonstrated the clear superiority of a proactive approach. The system achieved near-perfect stability by eliminating SLA violations entirely, while also attaining the highest task throughput of all tested models. The most efficient and well-rounded solution was found to be the AI's proactive scaling, which also lessened the latency trade-off present in the reactive TOC model. These results reinforces the main hypothesis of our paper, which is that an AI-driven and constraint-aware task scheduler outperforms both traditional and reactive constraint-based techniques in terms of stability, throughput and latency.

== Future Work

While this research validates the effectiveness of the Hybrid AI+TOC approach, several avenues for future work remain:

- *Online and Reinforcement Learning:* Offline training was used to train the current model.  The use of online learning or reinforcement learning agents that could continuously modify and enhance the prediction model in real-time based on live performance feedback may be investigated in future studies.

- *Heterogeneous and Multi-Cloud Environments:* The simulation was conducted with a homogeneous server pool. A valuable extension would be to evaluate the algorithm's performance in a more complex multi-cloud or edge computing or in a heterogeneous environment with servers of varying capacities.

- *Multi-Objective Optimization:* Our study predominantly optimized for stability and throughput. Future work could create a more comprehensive scheduling algorithm by incorporating additional objectives, such as operational cost or energy efficiency into the AI's decision-making process.

#bibliography("refs.bib", title: "References")
