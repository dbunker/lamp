d3(o3).
d2(o2).
d4(o2).
d0(o3).

d0(X) :- not d3(X), d2(X).
d0(X) :- not d3(X), d4(X).