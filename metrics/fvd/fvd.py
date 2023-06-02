# coding=utf-8

# Taken from https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
# with cosmetic modifications.
# See the copyright notice below and the license copy at the end of the file.

# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal Reference implementation for the Frechet Video Distance (FVD).

FVD is a metric for the quality of video generation models. It is inspired by
the FID (Frechet Inception Distance) used for images, but uses a different
embedding to be better suitable for videos.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub


def preprocess(videos, target_resolution, cast=True):
    """Runs some preprocessing on the videos for I3D model.

    Args:
    videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
        preprocessed. We don't care about the specific dtype of the videos, it can
        be anything that tf.image.resize_bilinear accepts. Values are expected to
        be in the range 0-255.
    target_resolution: (width, height): target video resolution

    Returns:
    videos: <float32>[batch_size, num_frames, height, width, depth]
    """
    videos_shape = videos.shape.as_list()
    all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
    # resized_videos = tf.image.resize_bilinear(all_frames, size=target_resolution)
    resized_videos = tf.image.resize(all_frames, size=target_resolution)
    target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
    output_videos = tf.reshape(resized_videos, target_shape)
    if cast:
        scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
        return scaled_videos
    else:
        return output_videos


def _is_in_graph(tensor_name):
    """Checks whether a given tensor does exists in the graph."""
    try:
        tf.get_default_graph().get_tensor_by_name(tensor_name)
    except KeyError:
        return False
    return True


def create_id3_embedding(videos):
    """Embeds the given videos using the Inflated 3D Convolution network.

    Downloads the graph of the I3D from tf.hub and adds it to the graph on the
    first call.

    Args:
        videos: <float32>[batch_size, num_frames, height=224, width=224, depth=3].
        Expected range is [-1, 1].

    Returns:
        embedding: <float32>[batch_size, embedding_size]. embedding_size depends
                on the model used.

    Raises:
        ValueError: when a provided embedding_layer is not supported.
    """

    batch_size = 16
    module_spec = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"

    # Making sure that we import the graph separately for
    # each different input video tensor.
    module_name = "fvd_kinetics-400_id3_module_" + videos.name.replace(":", "_")

    assert_ops = [
        tf.Assert(
            tf.reduce_max(videos) <= 1.001,
            ["max value in frame is > 1", videos]),
        tf.Assert(
            tf.reduce_min(videos) >= -1.001,
            ["min value in frame is < -1", videos]),
        tf.assert_equal(
            tf.shape(videos)[0],
            batch_size, ["invalid frame batch size: ", tf.shape(videos)],
            summarize=6),
    ]
    with tf.control_dependencies(assert_ops):
        videos = tf.identity(videos)

    module_scope = "%s_apply_default/" % module_name

    # To check whether the module has already been loaded into the graph, we look
    # for a given tensor name. If this tensor name exists, we assume the function
    # has been called before and the graph was imported. Otherwise we import it.
    # Note: in theory, the tensor could exist, but have wrong shapes.
    # This will happen if create_id3_embedding is called with a frames_placehoder
    # of wrong size/batch size, because even though that will throw a tf.Assert
    # on graph-execution time, it will insert the tensor (with wrong shape) into
    # the graph. This is why we need the following assert.
    video_batch_size = int(videos.shape[0])
    assert video_batch_size in [batch_size, -1, None], "Invalid batch size"
    tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
    if not _is_in_graph(tensor_name):
        i3d_model = hub.Module(module_spec, name=module_name)
        i3d_model(videos)

    # gets the kinetics-i3d-400-logits layer
    tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
    return tensor


def calculate_fvd(real_activations, generated_activations):
    """Returns a list of ops that compute metrics as funcs of activations.

    Args:
        real_activations: <float32>[num_samples, embedding_size]
        generated_activations: <float32>[num_samples, embedding_size]

    Returns:
        A scalar that contains the requested FVD.
    """
    return tf.contrib.gan.eval.frechet_classifier_distance_from_activations(
        real_activations, generated_activations)


#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/

#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

#    1. Definitions.

#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.

#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.

#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.

#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.

#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.

#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.

#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).

#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.

#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."

#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.

#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.

#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.

#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:

#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and

#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and

#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and

#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.

#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.

#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.

#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.

#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.

#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.

#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.

#    END OF TERMS AND CONDITIONS

#    APPENDIX: How to apply the Apache License to your work.

#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.

#    Copyright [yyyy] [name of copyright owner]

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
