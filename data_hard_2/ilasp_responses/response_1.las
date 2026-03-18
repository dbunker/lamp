%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.016s
%% Hypothesis Space Generation             : 3.44s
%% Conflict analysis                       : 30.976s
%%   - Positive Examples                   : 30.976s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 9.824s
%% Total                                   : 45.952s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d2(V1) :- d12(V1).
d7(V1) :- d2(V1).
d11(V1) :- d2(V1).
d5(V1) :- d2(V1).
d6(V1) :- d3(V1).

