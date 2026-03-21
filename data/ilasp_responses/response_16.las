%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 1.476s
%% Conflict analysis                       : 16.546s
%%   - Positive Examples                   : 16.546s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 6.239s
%% Total                                   : 25.06s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d4(V1) :- d3(V1).
d7(V1) :- d3(V1).
d1(V1) :- d3(V1).
d5(V1) :- d0(V1).
d4(V1) :- d0(V1).
d8(V1) :- d6(V1).
d2(V1) :- d4(V1); not d0(V1).

