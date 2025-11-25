d3(o3).
d3(o0).
d1(o3).
d0(o2).
d2(o2).

d1(X) :- not d0(X), d2(X).
d2(X) :- not d0(X), d3(X).