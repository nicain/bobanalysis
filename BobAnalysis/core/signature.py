import pandas as pd

session_signature_df_dict = {}
session_signature_df_dict['three_session_C'] = pd.DataFrame({'stimulus':
                                                                       ['locally_sparse_noise',
                                                                        'gap',
                                                                        'spontaneous',
                                                                        'gap',
                                                                        'natural_movie_one',
                                                                        'gap',
                                                                        'locally_sparse_noise',
                                                                        'gap',
                                                                        'natural_movie_two',
                                                                        'gap',
                                                                        'spontaneous',
                                                                        'gap',
                                                                        'locally_sparse_noise'],
                                                              'duration':[21724,
                                                                          150,
                                                                          8909,
                                                                          1,
                                                                          9050,
                                                                          905,
                                                                          21723,
                                                                          906,
                                                                          9051,
                                                                          150,
                                                                          8901,
                                                                          1,
                                                                          23534]})