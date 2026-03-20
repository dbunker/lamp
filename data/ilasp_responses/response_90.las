%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.192s
%% Conflict analysis                       : 0.566s
%%   - Positive Examples                   : 0.566s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.001s
%% Hypothesis Search                       : 0.129s
%% Total                                   : 0.93s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d4(V1) :- d0(V1).
d1(V1) :- d2(V1); not d4(V1).
d0(V1) :- d3(V1); not d2(V1).
d3(V1) :- d1(V1); not d5(V1).

