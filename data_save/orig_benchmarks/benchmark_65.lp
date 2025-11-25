d3(o3).
d2(o3).
d0(o2).
d0(o0).

d4(X) :- not d3(X), d0(X).
d1(X) :- not d0(X), d2(X).