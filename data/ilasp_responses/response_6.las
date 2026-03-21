%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.009s
%% Hypothesis Space Generation             : 2.481s
%% Conflict analysis                       : 44.371s
%%   - Positive Examples                   : 44.371s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 15.138s
%% Total                                   : 63.353s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d7(V1) :- d1(V1).
d6(V1) :- d7(V1).
d5(V1) :- d1(V1).
d4(V1) :- d8(V1).
d2(V1) :- d3(V1).
d3(V1) :- d4(V1).

