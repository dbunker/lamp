d2(o3).
d4(o2).
d0(o4).
d2(o1).

d3(X) :- d4(X), not d5(X).
d3(X) :- not d4(X), d2(X).