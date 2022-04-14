import pickle
import numpy as np
import sys
import datetime
import matplotlib.pyplot as plot


def analyze(files, verbose=False):
    accuracies, forgettings, epochtimes, diskusages, cpuusages, times = [], [], [], [], [], []
    for filename in files:
        data = pickle.load(open(filename, "rb"))
        result = data[0]
        time = data[1]
        try:
            if (verbose): print("\t Final accuracy:\t" + "{:.2f}%".format(result['finalacc']*100))
        except:
            pass
        if (verbose): print("\t Final loss:\t\t" + "{:.5f}".format(result['finalloss']))
        if (verbose): print("\t Loss: ", end="")
        for res in result['results']:
            if (verbose): print("{:.3f}".format(res['loss']), end=" -> ")
        if (verbose): print("end")
        if (verbose): print("\t Acc: ", end="")
        for res in result['results']:
            if (verbose): print("{:.2f}%".format(res['acc']), end=" -> ")
        l = [line['acc'] for line in result['results']]
        if (verbose): print("end")
        accuracies.extend(l)

        if (verbose): print("\t Forg: ", end="")
        for res in result['results']:
            if (verbose): print("{:.3f}".format(res['forg']), end=" -> ")
        l = [line['forg'] for line in result['results']]
        if (verbose): print("end")
        forgettings.extend(l)

        l = []
        if (verbose): print("\t Epoch times: ", end="")
        res = result['results'][-1]["all"]
        for key, value in res.items():
            if (key.startswith("Time_Epoch/train_phase")):
                l.append(value)
                if (verbose): print("{:.3f}".format(value), end=" -> ")
        if(verbose): print("end")
        epochtimes.extend(l)

        l = []
        if (verbose): print("\t Disk usage: ", end="")
        res = result['results'][-1]["all"]
        for key, value in res.items():
            if (key.startswith("DiskUsage_Epoch/train_phase")):
                l.append(value)
                if (verbose): print("{:.3f}".format(value), end=" -> ")
        if(verbose): print("end")
        diskusages.extend(l)

        times.append(time["time"])
        if (verbose): print("\t Time: " + "{:.5f}s".format(time["time"]))
        if (verbose): print("\n")

    print("\t Epoch times (" + str(len(epochtimes)) + ") " + "{:.3f}".format(np.mean(epochtimes)) + " +- " + "{:.3f}".format(np.std(epochtimes)))
    print("\t Accuracy (" + str(len(accuracies)) + ") " + "{:.2f}%".format(np.mean(accuracies)) + " +- " + "{:.2f}%".format(np.std(accuracies)))
    print("\t Forgetting (" + str(len(forgettings)) + ") " + "{:.3f}".format(np.mean(forgettings)) + " +- " + "{:.3f}".format(np.std(forgettings)))
    print("\t Disk usage (" + str(len(diskusages)) + ") " + "{:.3f}".format(np.mean(diskusages)) + " +- " + "{:.3f}".format(np.std(diskusages)))
    print("\t Time (" + str(len(times)) + ") " + "{:.3f}".format(np.mean(times)) + " +- " + "{:.3f}".format(np.std(times)))
    print("\t\t Total time " + str(datetime.timedelta(seconds = np.sum(times))))
    print("\n\n")


def cumulative_stats(files):
    times = []
    for filename in files:
        data = pickle.load(open(filename, "rb"))
        times.append(data[1]["time"])
    print("Total time: " + "{:.5f}s".format(np.sum(times)) + " (" + str(datetime.timedelta(seconds = np.sum(times))) + ")")
    print("Number of trainings: " + str(len(files)))


def get_forgetting(files):
    forgettings = []
    for filename in files:
        data = pickle.load(open(filename, "rb"))
        result = data[0]
        l = [line['forg'] for line in result['results']]
        forgettings = l
    return (np.mean(forgettings), np.std(forgettings))


def get_times(files):
    epochtimes = []
    for filename in files:
        data = pickle.load(open(filename, "rb"))
        result = data[0]
        l = []
        res = result['results'][-1]["all"]
        for key, value in res.items():
            if (key.startswith("Time_Epoch/train_phase")):
                l.append(value)
        epochtimes = l
    return (np.mean(epochtimes), np.std(epochtimes))

plot_labels = []
plot_times = []
plot_forgettings = []
plot_times_err = []
plot_forgettings_err = []
colors = ["orange", "green", "red", "purple", "olive", "cyan"]
allfiles = []
print("WESAD offline")
files = []
for i in range(5):
    files.append("results/wesad_offline_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("Offline")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("WESAD naive")
files = []
for i in range(5):
    files.append("results/wesad_naive_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("Naive")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("WESAD replay")
files = []
for i in range(5):
    files.append("results/wesad_replay_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("Replay")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("WESAD cumulative")
files = []
for i in range(5):
    files.append("results/wesad_cumulative_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("Cumulative")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("WESAD EWC")
files = []
for i in range(5):
    files.append("results/wesad_ewc_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("EWC")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("WESAD LwF")
files = []
for i in range(5):
    files.append("results/wesad_lwf_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

fig, ax1 = plot.subplots()
ax1.set_title("WESAD - Epoch times and forgetting")
x_ticks = [2*i for i in range(len(plot_labels))]
ax2 = ax1.twinx()
ax1.set_xlabel('Strategies')
ax1.set_xticks(x_ticks)
ax1.set_ylabel('Epoch times')
ax1.bar(x_ticks, plot_times, yerr=plot_times_err, color="blue", width=-0.45, align="edge", tick_label=plot_labels)
ax2.set_xticks(x_ticks)
ax2.set_ylabel('Forgetting')
ax2.bar(x_ticks, plot_forgettings, yerr=plot_forgettings_err, color="red", width=0.45, align="edge", tick_label=plot_labels)
plot.savefig("WESAD.png")

plot_labels = []
plot_times = []
plot_forgettings = []
plot_times_err = []
plot_forgettings_err = []
print("ASCERTAIN offline")
files = []
for i in range(5):
    files.append("results/ascertain_offline_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("ASCERTAIN naive")
files = []
for i in range(5):
    files.append("results/ascertain_naive_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("ASCERTAIN replay")
files = []
for i in range(5):
    files.append("results/ascertain_replay_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("ASCERTAIN cumulative")
files = []
for i in range(5):
    files.append("results/ascertain_cumulative_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("ASCERTAIN EWC")
files = []
for i in range(5):
    files.append("results/ascertain_ewc_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("ASCERTAIN LwF")
files = []
for i in range(5):
    files.append("results/ascertain_lwf_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

fig, ax1 = plot.subplots()
ax1.set_title("ASCERTAIN - Epoch times and forgetting")
x_ticks = [2*i for i in range(len(plot_labels))]
ax2 = ax1.twinx()
ax1.set_xlabel('Strategies')
ax1.set_xticks(x_ticks)
ax1.set_ylabel('Epoch times')
ax1.bar(x_ticks, plot_times, yerr=plot_times_err, color="blue", width=-0.45, align="edge", tick_label=plot_labels)
ax2.set_xticks(x_ticks)
ax2.set_ylabel('Forgetting')
ax2.bar(x_ticks, plot_forgettings, yerr=plot_forgettings_err, color="red", width=0.45, align="edge", tick_label=plot_labels)
plot.savefig("ASCERTAIN.png")

plot_labels = []
plot_times = []
plot_forgettings = []
plot_times_err = []
plot_forgettings_err = []
print("Custom ASCERTAIN offline")
files = []
for i in range(5):
    files.append("results/customascertain_offline_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("Custom ASCERTAIN naive")
files = []
for i in range(5):
    files.append("results/customascertain_naive_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("Custom ASCERTAIN replay")
files = []
for i in range(5):
    files.append("results/customascertain_replay_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("Custom ASCERTAIN cumulative")
files = []
for i in range(5):
    files.append("results/customascertain_cumulative_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("Custom ASCERTAIN EWC")
files = []
for i in range(5):
    files.append("results/customascertain_ewc_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

print("Custom ASCERTAIN LwF")
files = []
for i in range(5):
    files.append("results/customascertain_lwf_results" + str(i) + ".pkl")
allfiles.extend(files)
analyze(files)
plot_labels.append("LwF")
plot_times.append(get_times(files)[0])
plot_forgettings.append(get_forgetting(files)[0])
plot_times_err.append(get_times(files)[1])
plot_forgettings_err.append(get_forgetting(files)[1])

fig, ax1 = plot.subplots()
ax1.set_title("Custom ASCERTAIN - Epoch times and forgetting")
x_ticks = [2*i for i in range(len(plot_labels))]
ax2 = ax1.twinx()
ax1.set_xlabel('Strategies')
ax1.set_xticks(x_ticks)
ax1.set_ylabel('Epoch times')
ax1.bar(x_ticks, plot_times, yerr=plot_times_err, color="blue", width=-0.45, align="edge", tick_label=plot_labels)
ax2.set_xticks(x_ticks)
ax2.set_ylabel('Forgetting')
ax2.bar(x_ticks, plot_forgettings, yerr=plot_forgettings_err, color="red", width=0.45, align="edge", tick_label=plot_labels)
plot.savefig("CustomASCERTAIN.png")

cumulative_stats(allfiles)
