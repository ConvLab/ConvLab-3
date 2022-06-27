import tensorflow as tf


class TF_Writer():

    def __init__(self, log_dir):
        self.set_tf_writer(log_dir)

    def set_tf_writer(self, log_dir):
        with tf.device('cpu:0'):
            self.sess = tf.Session()
            self.tf_writer = tf.summary.FileWriter(log_dir)
            self.complete_rate = tf.placeholder(tf.float32, shape=[])
            self.success_rate = tf.placeholder(tf.float32, shape=[])
            self.avg_turns = tf.placeholder(tf.float32, shape=[])
            self.avg_return = tf.placeholder(tf.float32, shape=[])
            self.complete_rate_summary = tf.summary.scalar(name='complete_rate', tensor=self.complete_rate)
            self.success_rate_summary = tf.summary.scalar(name='success_rate', tensor=self.success_rate)
            self.avg_turns_summary = tf.summary.scalar(name='avg_turns', tensor=self.avg_turns)
            self.avg_return_summary = tf.summary.scalar(name='avg_return', tensor=self.avg_return)
            self.merged_summary = tf.summary.merge_all()

    def write_summary(self, complete_rate, success_rate, avg_turns, avg_return, counter):
        with tf.device('cpu:0'):
            feed_dict = {self.complete_rate: complete_rate, self.success_rate: success_rate, self.avg_turns: avg_turns,
                         self.avg_return: avg_return}

            summary = self.sess.run(self.merged_summary, feed_dict=feed_dict)
            self.tf_writer.add_summary(summary, counter)