%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 1.376s
%% Conflict analysis                       : 50.953s
%%   - Positive Examples                   : 50.953s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 20.653s
%% Total                                   : 73.794s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d1(V1) :- d8(V1).
d2(V1) :- d7(V1).
d3(V1) :- d7(V1).
d8(V1) :- d7(V1).
d9(V1) :- d5(V1).
d3(V1) :- d4(V1).
d0(V1) :- d2(V1); not d7(V1).

