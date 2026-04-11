d0(o0).
d4(o4).
d5(o1).
d4(o1).
d3(o4).
d4(o3).

d3(X) :- d4(X).
d3(X) :- d0(X).
d2(X) :- not d5(X), d0(X).
d0(X) :- d5(X).